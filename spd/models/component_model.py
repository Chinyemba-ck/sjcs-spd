from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, get_args, get_origin
from typing_extensions import override

import torch
import wandb
import yaml
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.pytorch_utils import Conv1D as RadfordConv1D
from wandb.apis.public import Run

from spd.configs import Config
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.components import (
    Components,
    ComponentsMaskInfo,
    EmbeddingComponents,
    GateType,
    Identity,
    LinearComponents,
    MLPGates,
    VectorMLPGates,
    VectorSharedMLPGate,
)
from spd.models.sigmoids import SIGMOID_TYPES, SigmoidTypes
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.general_utils import fetch_latest_local_checkpoint, resolve_class
from spd.utils.module_utils import get_target_module_paths
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


def _get_default_for_field(field_name: str, field_info, config_data: dict) -> Any:
    """Generate intelligent default value for a missing required field.

    Args:
        field_name: Name of the field that needs a default
        field_info: Pydantic FieldInfo object containing field metadata
        config_data: Existing config data to use for context-aware defaults

    Returns:
        A reasonable default value for the field

    Raises:
        ValueError: If cannot generate a sensible default for the field type
    """
    # Special cases for known fields based on their semantics
    if field_name == 'n_examples_until_dead':
        # Calculate based on existing config values
        train_log_freq = config_data.get('train_log_freq', 100)
        batch_size = config_data.get('batch_size', 32)
        return train_log_freq * batch_size

    # Get field type from annotation
    field_type = field_info.annotation
    type_origin = get_origin(field_type)
    type_str = str(field_type)

    # Handle common types and their constraints
    if 'PositiveInt' in type_str or (field_type == int and 'positive' in field_name.lower()):
        # Positive integer fields
        if 'freq' in field_name or 'frequency' in field_name:
            return 100
        elif 'count' in field_name or 'num' in field_name or 'n_' in field_name:
            return 10
        elif 'size' in field_name:
            return 32
        elif 'samples' in field_name:
            return 10
        else:
            return 1

    elif 'NonNegativeInt' in type_str or field_type == int:
        # Non-negative or regular integer fields
        if 'freq' in field_name or 'frequency' in field_name:
            return 100
        elif 'count' in field_name or 'num' in field_name:
            return 10
        elif 'size' in field_name:
            return 32
        elif 'seed' in field_name:
            return 0
        else:
            return 0

    elif field_type == float or 'Float' in type_str:
        # Float fields
        if 'rate' in field_name or 'lr' in field_name:
            return 0.001
        elif 'threshold' in field_name:
            return 0.5
        elif 'epsilon' in field_name or 'eps' in field_name:
            return 1e-8
        else:
            return 0.0

    elif field_type == str:
        # String fields
        if 'name' in field_name:
            return "default"
        elif 'path' in field_name:
            return ""
        else:
            return ""

    elif field_type == bool:
        # Boolean fields - default to False for safety
        return False

    elif type_origin == list:
        # List fields - empty list
        return []

    elif type_origin == dict:
        # Dict fields - empty dict
        return {}

    elif type_origin == tuple:
        # Tuple fields - empty tuple
        return ()

    elif type_origin == set:
        # Set fields - empty set
        return set()

    elif 'Literal' in type_str:
        # For Literal types, try to extract the first allowed value
        args = get_args(field_type)
        if args:
            return args[0]
        else:
            raise ValueError(f"Cannot determine default for Literal field '{field_name}' with no args")

    elif type_origin is not None and hasattr(type_origin, '__mro__'):
        # For custom types, check if they have defaults or can be instantiated
        # This is a last resort and may not work for all types
        try:
            # Try to instantiate with no arguments
            return type_origin()
        except:
            pass

    # If we can't determine a sensible default, raise an error
    # This is better than silently using a wrong default
    raise ValueError(
        f"Cannot generate default for field '{field_name}' of type {field_type}. "
        f"Please provide a value in the config or update the default generation logic."
    )


@dataclass
class SPDRunInfo(RunInfo):
    """Run info from training a ComponentModel (i.e. from an SPD run)."""

    @override
    @classmethod
    def from_path(cls, path: ModelPath) -> "SPDRunInfo":
        """Load the run info from a wandb run or a local path to a checkpoint."""
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                comp_model_path = fetch_latest_local_checkpoint(run_dir, prefix="model")
                config_path = run_dir / "final_config.yaml"
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                comp_model_path, config_path = ComponentModel._download_wandb_files(wandb_path)
        else:
            path = Path(path)
            # Find any YAML config file in the directory
            yaml_files = list(path.glob("*.yaml"))
            
            if yaml_files:
                # Directory contains config files - this is likely the files directory
                # Prefer final_config.yaml if it exists, otherwise use any yaml file
                config_candidates = [f for f in yaml_files if f.name == "final_config.yaml"]
                config_path = config_candidates[0] if config_candidates else yaml_files[0]
                
                # Find the model checkpoint file
                model_files = list(path.glob("*.pth")) + list(path.glob("*.pt")) + list(path.glob("*.bin"))
                if model_files:
                    comp_model_path = model_files[0]
                else:
                    comp_model_path = path
            else:
                # No yaml files in current dir, check parent (regular SPD output structure)
                comp_model_path = path
                parent_yaml = list(path.parent.glob("*.yaml"))
                if not parent_yaml:
                    raise ValueError(f"No config files found in {path} or its parent directory")
                config_path = parent_yaml[0]

        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            
            # Check if this is W&B sweep format (values wrapped in {value: ...} dicts)
            # by checking if any top-level values are dicts with 'value' key
            is_sweep_format = any(
                isinstance(v, dict) and 'value' in v 
                for v in config_data.values() 
                if v is not None
            )
            
            if is_sweep_format:
                # Unwrap W&B sweep format
                unwrapped_data = {}
                for key, val in config_data.items():
                    if isinstance(val, dict) and 'value' in val:
                        unwrapped_data[key] = val['value']
                    else:
                        # Keep as-is if not in sweep format (e.g., _wandb metadata)
                        if not key.startswith('_'):  # Skip metadata keys
                            unwrapped_data[key] = val
                config_data = unwrapped_data

            # General backward compatibility handler
            # Dynamically detect and handle schema changes

            # Get all fields defined in current Config model
            current_fields = set(Config.model_fields.keys())
            config_fields = set(config_data.keys())

            # Detect and remove deprecated fields (fields in config that aren't in model)
            deprecated_fields = config_fields - current_fields
            for field in deprecated_fields:
                logger.info(f"Removing deprecated config field: {field} (value was: {config_data[field]})")
                del config_data[field]

            # Detect missing required fields and add intelligent defaults
            missing_fields = current_fields - config_fields
            for field_name in missing_fields:
                field_info = Config.model_fields[field_name]

                # Check if field is required (has no default)
                if field_info.is_required():
                    try:
                        # Generate an intelligent default based on field characteristics
                        default_value = _get_default_for_field(field_name, field_info, config_data)
                        logger.info(f"Adding missing required field '{field_name}' with default value: {default_value}")
                        config_data[field_name] = default_value
                    except ValueError as e:
                        # If we can't generate a default, log the error and re-raise
                        logger.error(f"Failed to generate default for field '{field_name}': {e}")
                        raise
                elif field_info.default is not None:
                    # Use the model's default if available
                    logger.info(f"Adding missing optional field '{field_name}' with model default: {field_info.default}")
                    config_data[field_name] = field_info.default
                elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                    # Use default factory if available
                    default_value = field_info.default_factory()
                    logger.info(f"Adding missing optional field '{field_name}' with factory default: {default_value}")
                    config_data[field_name] = default_value

            config = Config(**config_data)

        return cls(checkpoint_path=comp_model_path, config=config)


class ComponentModel(LoadableModule):
    """Wrapper around an arbitrary pytorch model for running SPD.

    The underlying *base model* can be any subclass of `nn.Module` (e.g.
    `LlamaForCausalLM`, `AutoModelForCausalLM`) as long as its sub-module names
    match the patterns you pass in `target_module_patterns`.

    Forward passes are performed in three modes:
    - 'target': Standard forward pass of the target model
    - 'components': Forward with (masked) components replacing chosen modules. The components are
        inserted in place of the chosen modules with the use of forward hooks.
    - 'input_cache': Forward with caching inputs to chosen modules

    We register components and gates as modules in this class in order to have them update
    correctly when the model is wrapped in a `DistributedDataParallel` wrapper (and for other
    conveniences).
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_module_patterns: list[str],
        C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
        pretrained_model_output_attr: str | None,
    ):
        super().__init__()

        for name, param in target_model.named_parameters():
            assert not param.requires_grad, (
                f"Target model should not have any trainable parameters. "
                f"Found {param.requires_grad} for {name}"
            )

        self.target_model = target_model
        self.C = C
        self.pretrained_model_output_attr = pretrained_model_output_attr

        module_paths = get_target_module_paths(target_model, target_module_patterns)

        self.components = ComponentModel._create_components(
            target_model=target_model,
            module_paths=module_paths,
            C=C,
        )
        self._components = nn.ModuleDict(
            {k.replace(".", "-"): self.components[k] for k in sorted(self.components)}
        )

        self.gates = ComponentModel._create_gates(
            target_model=target_model,
            module_paths=module_paths,
            C=C,
            gate_type=gate_type,
            gate_hidden_dims=gate_hidden_dims,
        )
        self._gates = nn.ModuleDict(
            {k.replace(".", "-"): self.gates[k] for k in sorted(self.gates)}
        )

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        target_module = self.target_model.get_submodule(module_name)

        match target_module:
            case RadfordConv1D():
                return target_module.weight.T
            case nn.Linear() | nn.Embedding():
                return target_module.weight
            case Identity():
                p = next(self.parameters())
                return torch.eye(target_module.d, device=p.device, dtype=p.dtype)
            case _:
                raise ValueError(f"Module {target_module} not supported")

    @staticmethod
    def _create_component(
        target_module: nn.Module,
        C: int,
    ) -> Components:
        match target_module:
            case nn.Linear():
                d_out, d_in = target_module.weight.shape
                component = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case RadfordConv1D():
                d_in, d_out = target_module.weight.shape
                component = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case Identity():
                component = LinearComponents(
                    C=C,
                    d_in=target_module.d,
                    d_out=target_module.d,
                    bias=None,
                )
            case nn.Embedding():
                component = EmbeddingComponents(
                    C=C,
                    vocab_size=target_module.num_embeddings,
                    embedding_dim=target_module.embedding_dim,
                )
            case _:
                raise ValueError(f"Module {target_module} not supported")

        return component

    @staticmethod
    def _create_components(
        target_model: nn.Module,
        module_paths: list[str],
        C: int,
    ) -> dict[str, Components]:
        components: dict[str, Components] = {}
        for module_path in module_paths:
            target_module = target_model.get_submodule(module_path)
            components[module_path] = ComponentModel._create_component(target_module, C)
        return components

    @staticmethod
    def _create_gate(
        target_module: nn.Module,
        component_C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
    ) -> nn.Module:
        """Helper to create a gate based on gate_type and module type."""
        if isinstance(target_module, nn.Embedding):
            assert gate_type == "mlp", "Embedding modules only supported for gate_type='mlp'"

        if gate_type == "mlp":
            return MLPGates(C=component_C, hidden_dims=gate_hidden_dims)

        match target_module:
            case nn.Linear():
                input_dim = target_module.weight.shape[1]
            case RadfordConv1D():
                input_dim = target_module.weight.shape[0]
            case Identity():
                input_dim = target_module.d
            case _:
                raise ValueError(f"Module {type(target_module)} not supported for {gate_type=}")

        match gate_type:
            case "vector_mlp":
                return VectorMLPGates(
                    C=component_C, input_dim=input_dim, hidden_dims=gate_hidden_dims
                )
            case "shared_mlp":
                return VectorSharedMLPGate(
                    C=component_C, input_dim=input_dim, hidden_dims=gate_hidden_dims
                )

    @staticmethod
    def _create_gates(
        target_model: nn.Module,
        module_paths: list[str],
        C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
    ) -> dict[str, nn.Module]:
        gates: dict[str, nn.Module] = {}
        for module_path in module_paths:
            target_module = target_model.get_submodule(module_path)
            gates[module_path] = ComponentModel._create_gate(
                target_module,
                C,
                gate_type,
                gate_hidden_dims,
            )
        return gates

    def _extract_output(self, raw_output: Any) -> Any:
        """Extract the desired output from the model's raw output.

        If pretrained_model_output_attr is None, returns the raw output directly.
        If pretrained_model_output_attr starts with "idx_", returns the index specified by the
        second part of the string. E.g. "idx_0" returns the first element of the raw output.
        Otherwise, returns the specified attribute from the raw output.

        Args:
            raw_output: The raw output from the model.

        Returns:
            The extracted output.
        """
        if self.pretrained_model_output_attr is None:
            return raw_output
        elif self.pretrained_model_output_attr.startswith("idx_"):
            idx_val = int(self.pretrained_model_output_attr.split("_")[1])
            assert isinstance(raw_output, Sequence), (
                f"raw_output must be a sequence, not {type(raw_output)}"
            )
            assert idx_val < len(raw_output), (
                f"Index {idx_val} out of range for raw_output of length {len(raw_output)}"
            )
            return raw_output[idx_val]
        else:
            return getattr(raw_output, self.pretrained_model_output_attr)

    @override
    def forward(
        self,
        *args: Any,
        mode: Literal["target", "components", "input_cache"] | None = "target",
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        module_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass of the patched model.

        NOTE: We need all the forward options in this method in order for DistributedDataParallel to
        work (https://discuss.pytorch.org/t/is-it-ok-to-use-methods-other-than-forward-in-ddp/176509).

        Args:
            mode: The type of forward pass to perform:
                - 'target': Standard forward pass of the target model
                - 'components': Forward with component replacements (requires masks)
                - 'input_cache': Forward with input caching (requires module_names)
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo
                (required for mode='components').
            module_names: List of module names to cache inputs for
                (required for mode='input_cache')

        If `pretrained_model_output_attr` is set, return the attribute of the model's output.
        """
        match mode:
            case "components":
                assert mask_infos is not None, "mask_infos are required for mode='components'"
                return self._forward_with_components(*args, mask_infos=mask_infos, **kwargs)
            case "input_cache":
                assert module_names is not None, (
                    "module_names parameter is required for mode='input_cache'"
                )
                return self._forward_with_input_cache(*args, module_names=module_names, **kwargs)
            case "target" | None:
                return self._extract_output(self.target_model(*args, **kwargs))

    @contextmanager
    def _attach_forward_hooks(
        self, hooks: dict[str, Callable[..., Any]]
    ) -> Generator[None, None, None]:
        """Context manager to temporarily attach forward hooks to the target model."""
        handles: list[RemovableHandle] = []
        for module_name, hook in hooks.items():
            target_module = self.target_model.get_submodule(module_name)
            handle = target_module.register_forward_hook(hook, with_kwargs=True)
            handles.append(handle)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _forward_with_components(
        self, *args: Any, mask_infos: dict[str, ComponentsMaskInfo], **kwargs: Any
    ) -> Any:
        """Forward pass with temporary component replacements. `masks` is a dictionary mapping
        component paths to mask infos. A mask info being present means that the module will be replaced
        with components, and the value of the mask info will be used as the mask for the components.

        Args:
            mask_infos: Dictionary mapping module names to ComponentsMaskInfo
        """

        def fwd_hook(
            _module: nn.Module,
            args: list[Any],
            kwargs: dict[Any, Any],
            output: Any,
            components: Components,
            mask_info: ComponentsMaskInfo,
        ) -> None | Any:
            assert len(args) == 1, "Expected 1 argument"
            assert len(kwargs) == 0, "Expected no keyword arguments"
            x = args[0]
            assert isinstance(x, Tensor), "Expected input tensor"
            assert isinstance(output, Tensor), (
                "Only supports single-tensor outputs, got type(output)"
            )

            components_out = components(
                x,
                mask=mask_info.component_mask,
                weight_delta_and_mask=mask_info.weight_delta_and_mask,
            )
            if mask_info.routing_mask is not None:
                return torch.where(mask_info.routing_mask[..., None], components_out, output)

            return components_out

        hooks: dict[str, Callable[..., Any]] = {}
        for module_name, mask_info in mask_infos.items():
            components = self.components[module_name]
            hooks[module_name] = partial(fwd_hook, components=components, mask_info=mask_info)

        with self._attach_forward_hooks(hooks):
            raw_out = self.target_model(*args, **kwargs)

        return self._extract_output(raw_out)

    def _forward_with_input_cache(
        self, *args: Any, module_names: list[str], **kwargs: Any
    ) -> tuple[Any, dict[str, Tensor]]:
        """Forward pass with caching at the input to the modules given by `module_names`.
        Args:
            module_names: List of module names to cache the inputs to.
        Returns:
            Tuple of (model output, input cache dictionary)
        """

        cache = {}

        def cache_hook(
            _module: nn.Module,
            args: list[Any],
            kwargs: dict[Any, Any],
            _output: Any,
            param_name: str,
        ) -> None:
            assert len(args) == 1, "Expected 1 argument"
            assert len(kwargs) == 0, "Expected no keyword arguments"
            x = args[0]
            assert isinstance(x, Tensor), "Expected x to be a tensor"
            cache[param_name] = x

        hooks = {
            module_name: partial(cache_hook, param_name=module_name) for module_name in module_names
        }
        with self._attach_forward_hooks(hooks):
            raw_out = self.target_model(*args, **kwargs)

        out = self._extract_output(raw_out)
        return out, cache

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> tuple[Path, Path]:
        """Download the relevant files from a wandb run.

        Returns:
            Tuple of (model_path, config_path)
        """
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        # Get any .pth or .pt checkpoint file (no specific prefix required)
        checkpoint = fetch_latest_wandb_checkpoint(run, prefix=None)

        run_dir = fetch_wandb_run_dir(run.id)

        # Try to find a config file - could be named differently in different runs
        config_files = [f.name for f in run.files() if f.name.endswith(".yaml")]
        if not config_files:
            raise ValueError(f"No config files found in run {wandb_project_run_id}")
        
        # Prefer final_config.yaml if it exists, otherwise take any yaml file
        config_file = "final_config.yaml" if "final_config.yaml" in config_files else config_files[0]
        
        config_path = download_wandb_file(run, run_dir, config_file)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)

        return checkpoint_path, config_path

    @classmethod
    def _convert_old_state_dict(cls, old_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert old checkpoint format to new format for backwards compatibility."""
        new_state_dict = {}

        for key, value in old_state_dict.items():
            # Convert model weights
            if key.startswith("model."):
                # model.linear1.weight -> patched_model.linear1.original.weight
                # model.linear1.bias -> patched_model.linear1.original.bias
                new_key = key.replace("model.", "patched_model.")
                parts = new_key.split(".")
                # Insert "original" before weight/bias
                if parts[-1] in ["weight", "bias"]:
                    parts.insert(-1, "original")
                    new_key = ".".join(parts)
                new_state_dict[new_key] = value

            # Convert component weights
            elif key.startswith("components."):
                # components.linear1.A -> patched_model.linear1.components.V
                # components.linear1.B -> patched_model.linear1.components.U
                # components.linear1.bias -> patched_model.linear1.components.bias
                module_name = key.split(".")[1]  # e.g., "linear1" or "hidden_layers-0"
                param_name = key.split(".")[2]   # e.g., "A", "B", or "bias"

                if param_name == "A":
                    new_key = f"patched_model.{module_name.replace('-', '.')}.components.V"
                elif param_name == "B":
                    new_key = f"patched_model.{module_name.replace('-', '.')}.components.U"
                else:  # bias
                    new_key = f"patched_model.{module_name.replace('-', '.')}.components.bias"
                new_state_dict[new_key] = value

            # Convert gate weights
            elif key.startswith("gates."):
                # gates.linear1.mlp_in -> _gates.linear1.layers.0.W
                # gates.linear1.in_bias -> _gates.linear1.layers.0.b
                # gates.linear1.mlp_out -> _gates.linear1.layers.2.W
                # gates.linear1.out_bias -> _gates.linear1.layers.2.b
                parts = key.split(".")
                module_name = parts[1]  # e.g., "linear1" or "hidden_layers-0"
                param_name = parts[2]   # e.g., "mlp_in", "in_bias", "mlp_out", "out_bias"

                if param_name == "mlp_in":
                    new_key = f"_gates.{module_name}.layers.0.W"
                elif param_name == "in_bias":
                    new_key = f"_gates.{module_name}.layers.0.b"
                elif param_name == "mlp_out":
                    new_key = f"_gates.{module_name}.layers.2.W"
                elif param_name == "out_bias":
                    new_key = f"_gates.{module_name}.layers.2.b"
                else:
                    # Unknown gate parameter, keep as is
                    new_key = key.replace("gates.", "_gates.")
                new_state_dict[new_key] = value
            else:
                # Keep any other keys as-is
                new_state_dict[key] = value

        return new_state_dict

    @classmethod
    def from_run_info(cls, run_info: RunInfo) -> "ComponentModel":
        """Load a trained ComponentModel checkpoint from a run info object."""
        config = run_info.config

        # Load the target model
        model_class = resolve_class(config.pretrained_model_class)
        if config.pretrained_model_name is not None:
            assert hasattr(model_class, "from_pretrained"), (
                f"Model class {model_class} should have a `from_pretrained` method"
            )
            target_model = model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            assert issubclass(model_class, LoadableModule), (
                f"Model class {model_class} should be a subclass of LoadableModule which "
                "defines a `from_pretrained` method"
            )
            assert run_info.config.pretrained_model_path is not None
            target_model = model_class.from_pretrained(run_info.config.pretrained_model_path)

        target_model.eval()
        target_model.requires_grad_(False)

        if config.identity_module_patterns is not None:
            insert_identity_operations_(
                target_model, identity_patterns=config.identity_module_patterns
            )

        comp_model = ComponentModel(
            target_model=target_model,
            target_module_patterns=config.all_module_patterns,
            C=config.C,
            gate_hidden_dims=config.gate_hidden_dims,
            gate_type=config.gate_type,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
        )

        comp_model_weights = torch.load(
            run_info.checkpoint_path, map_location="cpu", weights_only=True
        )

        handle_deprecated_state_dict_keys_(comp_model_weights)

        comp_model.load_state_dict(comp_model_weights)
        return comp_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "ComponentModel":
        """Load a trained ComponentModel checkpoint from a local or wandb path."""
        run_info = SPDRunInfo.from_path(path)
        return cls.from_run_info(run_info)

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sigmoid_type: SigmoidTypes,
        sampling: Literal["continuous", "binomial"],
        detach_inputs: bool = False,
    ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
        """Calculate causal importances.

        Args:
            pre_weight_acts: The activations before each layer in the target model.
            sigmoid_type: Type of sigmoid to use.
            detach_inputs: Whether to detach the inputs to the gates.

        Returns:
            Tuple of (causal_importances, causal_importances_upper_leaky) dictionaries for each layer.
        """
        causal_importances = {}
        causal_importances_upper_leaky = {}

        for param_name in pre_weight_acts:
            acts = pre_weight_acts[param_name]
            gates = self.gates[param_name]

            match gates:
                case MLPGates():
                    gate_input = self.components[param_name].get_inner_acts(acts)
                case VectorMLPGates() | VectorSharedMLPGate():
                    gate_input = acts
                case _:
                    raise ValueError(f"Unknown gate type: {type(gates)}")

            if detach_inputs:
                gate_input = gate_input.detach()

            gate_output = gates(gate_input)

            if sigmoid_type == "leaky_hard":
                lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
                upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
            else:
                # For other sigmoid types, use the same function for both
                lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
                upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

            gate_output_for_lower_leaky = gate_output
            if sampling == "binomial":
                gate_output_for_lower_leaky = 1.05 * gate_output - 0.05 * torch.rand_like(
                    gate_output
                )

            causal_importances[param_name] = lower_leaky_fn(gate_output_for_lower_leaky)
            causal_importances_upper_leaky[param_name] = upper_leaky_fn(gate_output).abs()

        return causal_importances, causal_importances_upper_leaky

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, " d_out d_in"]]:
        """Calculate the weight differences between the target and component weights (V@U) for each layer."""
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] = {}
        for comp_name, components in self.components.items():
            weight_deltas[comp_name] = self.target_weight(comp_name) - components.weight
        return weight_deltas


def _transform_key(key: str) -> str:
    key = key[:]  # make a copy

    # do this first to simplify the rest. All following logic assumes "target_model" naming convention
    key = key.replace("patched_model", "target_model")

    has_components = ".components." in key
    has_original = ".original." in key
    if has_components and has_original:
        raise ValueError(
            f"Key {key} has both components and original, this is technically possible but unsupported"
        )

    if has_components:
        # we need to move the path out of the target model, remove the "components" nesting,
        # normalize the path, and put it under the "_components" ModuleDict
        assert key.startswith("target_model.")
        key = key.removeprefix("target_model.")
        path, contents = key.split(".components.")
        normalized_path = path.replace(".", "-")
        key = f"_components.{normalized_path}.{contents}"

    if has_original:
        # simpler: just collapse the nesting
        key = key.replace(".original.", ".")

    return key


def handle_deprecated_state_dict_keys_(state_dict: dict[str, Tensor]) -> None:
    """Maps deprecated state dict keys to new state dict keys
    old: used nested ComponentsOrModule wrappers
    new: uses a single ModuleDict for all components
    """
    for key in list(state_dict.keys()):
        new_key = _transform_key(key)
        if new_key == key:
            continue
        assert new_key not in state_dict, f"Renamed key {key} already exists in state_dict"
        state_dict[new_key] = state_dict.pop(key)
