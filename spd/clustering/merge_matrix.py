from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from jaxtyping import Bool, Int
from torch import Tensor


@dataclass(kw_only=True, slots=True)
class GroupMerge:
    """Canonical component-to-group assignment.

    `group_idxs` is a length-`n_components` integer tensor; entry `c`
    gives the group index (0 to `k_groups-1`) that contains component `c`.
    """

    group_idxs: Int[Tensor, " n_components"]
    k_groups: int
    _dtype: ClassVar[torch.dtype] = torch.bool
    old_to_new_idx: dict[int | None, int | None] | None = None

    @property
    def n_components(self) -> int:
        return int(self.group_idxs.shape[0])

    @property
    def components_per_group(self) -> Int[Tensor, " k_groups"]:
        return torch.bincount(self.group_idxs, minlength=self.k_groups)

    def validate(self, *, require_nonempty: bool = True) -> None:
        v_min: int = self.group_idxs.min().item()
        v_max: int = self.group_idxs.max().item()
        if v_min < 0 or v_max >= self.k_groups:
            raise ValueError("group indices out of range")

        if require_nonempty:
            has_empty_groups = self.components_per_group.eq(0).any().item()
            if has_empty_groups:
                raise ValueError("one or more groups are empty")

    def to_matrix(
        self, device: torch.device | None = None
    ) -> Bool[Tensor, "k_groups n_components"]:
        if device is None:
            device = self.group_idxs.device
        mat = torch.zeros((self.k_groups, self.n_components), dtype=self._dtype, device=device)
        mat[self.group_idxs, torch.arange(self.n_components, device=device)] = True
        return mat

    @classmethod
    def from_matrix(cls, mat: Bool[Tensor, "k_groups n_components"]) -> "GroupMerge":
        if mat.dtype is not torch.bool:
            raise TypeError("mat must have dtype bool")
        if not mat.sum(dim=0).eq(1).all():
            raise ValueError("each column must contain exactly one True")
        group_idxs = mat.argmax(dim=0).to(torch.int64)
        inst = cls(group_idxs=group_idxs, k_groups=int(mat.shape[0]))
        inst.validate(require_nonempty=False)
        return inst

    @classmethod
    def random(
        cls,
        n_components: int,
        k_groups: int,
        *,
        ensure_groups_nonempty: bool = False,
        device: torch.device | str = "cpu",
    ) -> "GroupMerge":
        if ensure_groups_nonempty and n_components < k_groups:
            raise ValueError("n_components must be >= k_groups when ensure_groups_nonempty is True")
        if ensure_groups_nonempty:
            base = torch.arange(k_groups, device=device)
            if n_components > k_groups:
                extra = torch.randint(0, k_groups, (n_components - k_groups,), device=device)
                group_idxs = torch.cat((base, extra))
                group_idxs = group_idxs[torch.randperm(n_components, device=device)]
            else:
                group_idxs = base
        else:
            group_idxs = torch.randint(0, k_groups, (n_components,), device=device)
        inst = cls(group_idxs=group_idxs, k_groups=k_groups)
        inst.validate(require_nonempty=ensure_groups_nonempty)
        return inst

    @classmethod
    def identity(cls, n_components: int) -> "GroupMerge":
        """Creates a GroupMerge where each component is its own group."""
        return cls(
            group_idxs=torch.arange(n_components, dtype=torch.int64),
            k_groups=n_components,
        )

    def merge_groups(self, group_a: int, group_b: int) -> "GroupMerge":
        """Merges two groups into one, returning a new GroupMerge."""
        if group_a < 0 or group_b < 0 or group_a >= self.k_groups or group_b >= self.k_groups:
            raise ValueError("group indices out of range")
        if group_a == group_b:
            raise ValueError("Cannot merge a group with itself")

        # make sure group_a is the smaller index
        if group_a > group_b:
            group_a, group_b = group_b, group_a

        # make a copy
        new_idxs = self.group_idxs.clone()
        # wherever its currently b, change it to a
        new_idxs[new_idxs == group_b] = group_a
        # wherever i currently above b, change it to i-1
        new_idxs[new_idxs > group_b] -= 1
        # create a new GroupMerge instance
        merged: GroupMerge = GroupMerge(group_idxs=new_idxs, k_groups=self.k_groups - 1)

        # create a mapping from old to new group indices
        # `None` as a key is for the new group that contains both a and b
        # values of a and b are mapped to `None` since they are merged
        old_to_new_idx: dict[int | None, int | None] = dict()
        for i in range(self.k_groups):
            if i in {group_a, group_b}:
                old_to_new_idx[i] = None
            elif i <= group_b:
                old_to_new_idx[i] = i
            else:
                old_to_new_idx[i] = i - 1
        old_to_new_idx[None] = group_a  # the new group index for the merged group

        # HACK: store the mapping in the instance for later use
        merged.old_to_new_idx = old_to_new_idx  # type: ignore[assignment]

        # validate the new instance
        # merged.validate(require_nonempty=True)
        return merged

    def all_downstream_merged(self) -> "BatchedGroupMerge":
        downstream: list[GroupMerge] = []
        idxs: list[tuple[int, int]] = []
        for i in range(self.k_groups):
            for j in range(i + 1, self.k_groups):
                downstream.append(self.merge_groups(i, j))
                idxs.append((i, j))

        return BatchedGroupMerge.from_list(
            merge_matrices=downstream,
            meta=[{"merge_pair": t} for t in idxs],
        )

    def plot(
        self,
        show: bool = True,
        figsize: tuple[int, int] = (10, 3),
        show_row_sums: bool | None = None,
        ax: "plt.Axes | None" = None,
        component_labels: list[str] | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        merge_matrix = self.to_matrix()
        k_groups, _ = merge_matrix.shape
        group_sizes = merge_matrix.sum(dim=1)

        if show_row_sums is None:
            show_row_sums = k_groups <= 20

        if ax is not None:
            show_row_sums = False  # don't show row sums if we have an ax to plot on
            ax_mat = ax
        else:
            if show_row_sums:
                fig, (ax_mat, ax_lbl) = plt.subplots(
                    1, 2, figsize=figsize, gridspec_kw={"width_ratios": [10, 1]}
                )
            else:
                fig, ax_mat = plt.subplots(figsize=figsize)

        ax_mat.imshow(merge_matrix.cpu(), aspect="auto", cmap="Blues", interpolation="nearest")
        ax_mat.set_xlabel("Components")
        ax_mat.set_ylabel("Groups")
        ax_mat.set_title("Merge Matrix")
        
        # Add component labeling if component labels are provided
        if component_labels is not None:
            # Import the function here to avoid circular imports
            from spd.clustering.activations import add_component_labeling
            add_component_labeling(ax_mat, component_labels, axis='x')

        if show_row_sums:
            ax_lbl.set_xlim(0, 1)
            ax_lbl.set_ylim(-0.5, k_groups - 0.5)
            ax_lbl.invert_yaxis()
            ax_lbl.set_title("Row Sums")
            ax_lbl.axis("off")
            for i, size in enumerate(group_sizes):
                ax_lbl.text(0.5, i, str(size.item()), va="center", ha="center", fontsize=12)

        plt.tight_layout()
        if show:
            plt.show()


@dataclass(slots=True)
class BatchedGroupMerge:
    """Batch of merge matrices.

    `group_idxs` has shape `(batch, n_components)`; each row holds the
    group index for every component in that matrix.
    """

    group_idxs: Int[Tensor, " batch n_components"]
    k_groups: Int[Tensor, " batch"]
    meta: list[dict] | None = None
    _dtype: ClassVar[torch.dtype] = torch.bool

    @property
    def batch_size(self) -> int:
        return int(self.group_idxs.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.group_idxs.shape[1])

    @property
    def k_groups_unique(self) -> int:
        """Returns the number of groups across all matrices, throws exception if they differ."""
        k_groups_set: set[int] = set(self.k_groups.tolist())
        if len(k_groups_set) != 1:
            raise ValueError("All matrices must have the same number of groups")
        return k_groups_set.pop()

    # def validate(self, *, require_nonempty: bool = True) -> None:
    #     v_min: Int[Tensor, ""]
    #     v_max:
    #     print(f"{v_min=}, {v_max=}")
    #     print(f"{type(v_min)=}, {type(v_max)=}")
    #     if v_min < 0 or v_max >= self.k_groups.m
    #         raise ValueError("group indices out of range")

    def to_matrix(
        self, device: torch.device | None = None
    ) -> Bool[Tensor, "batch k_groups n_components"]:
        if device is None:
            device = self.group_idxs.device
        k_groups_u: int = self.k_groups_unique
        mat = torch.nn.functional.one_hot(self.group_idxs, num_classes=k_groups_u)
        return mat.permute(0, 2, 1).to(device=device, dtype=self._dtype)

    @classmethod
    def from_matrix(cls, mat: Bool[Tensor, "batch k_groups n_components"]) -> "BatchedGroupMerge":
        if mat.dtype is not torch.bool:
            raise TypeError("mat must have dtype bool")
        if not mat.sum(dim=1).eq(1).all():
            raise ValueError("each column must have exactly one True per matrix")
        group_idxs = mat.argmax(dim=1).to(torch.int64)
        batch_size: int = int(mat.shape[0])
        inst = cls(
            group_idxs=group_idxs,
            k_groups=torch.full((batch_size,), int(mat.shape[1]), dtype=torch.int64),
        )
        # inst.validate(require_nonempty=False)
        return inst

    @classmethod
    def from_list(
        cls,
        merge_matrices: list[GroupMerge],
        meta: list[dict[str, Any]] | None = None,
    ) -> "BatchedGroupMerge":
        group_idxs = torch.stack([mm.group_idxs for mm in merge_matrices], dim=0)
        k_groups = torch.tensor([mm.k_groups for mm in merge_matrices], dtype=torch.int64)
        inst = cls(group_idxs=group_idxs, k_groups=k_groups, meta=meta)
        # inst.validate(require_nonempty=False)
        return inst

    def __getitem__(self, idx: int) -> GroupMerge:
        if not (0 <= idx < self.batch_size):
            raise IndexError("index out of range")
        group_idxs = self.group_idxs[idx]
        k_groups: int = int(self.k_groups[idx].item())
        return GroupMerge(group_idxs=group_idxs, k_groups=k_groups)

    def __iter__(self):
        """Iterate over the GroupMerge instances in the batch."""
        for i in range(self.batch_size):
            yield self[i]

    def __len__(self) -> int:
        return self.batch_size

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape of the merge matrices as (batch_size, n_components)."""
        return self.batch_size, self.n_components

    @classmethod
    def random(
        cls,
        batch_size: int,
        n_components: int,
        k_groups: int,
        *,
        ensure_groups_nonempty: bool = False,
        device: torch.device | str = "cpu",
    ) -> "BatchedGroupMerge":
        return cls.from_list(
            [
                GroupMerge.random(
                    n_components=n_components,
                    k_groups=k_groups,
                    ensure_groups_nonempty=ensure_groups_nonempty,
                    device=device,
                )
                for _ in range(batch_size)
            ]
        )
