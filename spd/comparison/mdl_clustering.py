"""MDL clustering wrapper for clustering comparison interface."""

import time
import torch
import streamlit as st
from typing import Dict, List, Tuple, Any
from jaxtyping import Float
from torch import Tensor

from spd.clustering.activations import process_activations, ProcessedActivations
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory


class MDLClusteringConfig:
    """Configuration for MDL clustering that matches the interface config."""
    
    def __init__(self, **kwargs):
        # Default values from resid_mlp1.json
        self.activation_threshold = kwargs.get("activation_threshold", 0.01)
        self.alpha = kwargs.get("alpha", 1)
        self.iters = kwargs.get("iters", 140)
        self.merge_pair_sampling_method = kwargs.get("merge_pair_sampling_method", "range")
        self.merge_pair_sampling_kwargs = kwargs.get("merge_pair_sampling_kwargs", {"threshold": 0.05})
        self.pop_component_prob = kwargs.get("pop_component_prob", 0)
        self.filter_dead_threshold = kwargs.get("filter_dead_threshold", 0.01)
        
    def to_merge_config(self) -> MergeConfig:
        """Convert to MergeConfig object."""
        return MergeConfig(
            activation_threshold=self.activation_threshold,
            alpha=self.alpha,
            iters=self.iters,
            merge_pair_sampling_method=self.merge_pair_sampling_method,
            merge_pair_sampling_kwargs=self.merge_pair_sampling_kwargs,
            pop_component_prob=self.pop_component_prob,
        )


class MDLClusteringResults:
    """Results from MDL clustering."""
    
    def __init__(
        self,
        merge_history: MergeHistory,
        processed_activations: ProcessedActivations,
        timing_info: Dict[str, float],
        config: MDLClusteringConfig
    ):
        self.merge_history = merge_history
        self.processed_activations = processed_activations
        self.timing_info = timing_info
        self.config = config
        
    @property
    def final_groups(self) -> Dict[int, List[str]]:
        """Get final clustering groups."""
        if self.merge_history.n_iters_current == 0:
            return {}

        # Fixed: Use proper indexing instead of non-existent get_iteration method
        final_iter = self.merge_history[self.merge_history.n_iters_current - 1]
        final_merge = final_iter["merges"]  # Fixed: correct key is "merges" not "current_merge"
        groups = {}

        for component_idx, group_id in enumerate(final_merge.group_idxs):
            group_id_int = int(group_id.item())
            if group_id_int not in groups:
                groups[group_id_int] = []

            if component_idx < len(self.processed_activations.labels):
                groups[group_id_int].append(self.processed_activations.labels[component_idx])

        return groups

    @property
    def final_mdl_cost(self) -> float | None:
        """Get final MDL cost.

        Note: MDL cost is not stored in merge history for memory efficiency.
        The clustering is performed correctly using MDL minimization,
        but the final cost value is not available for display.
        """
        return None
    
    @property
    def num_final_groups(self) -> int:
        """Get number of final groups."""
        return len(self.final_groups)


@st.cache_data
def run_mdl_clustering(
    _activations_dict: Dict[str, Float[Tensor, "n_steps C"]],
    config_dict: Dict[str, Any]
) -> MDLClusteringResults:
    """Run MDL clustering on component activations.

    Args:
        _activations_dict: Dictionary of activations per module (underscore for Streamlit caching)
        config_dict: Configuration dictionary

    Returns:
        MDLClusteringResults object
    """
    config = MDLClusteringConfig(**config_dict)
    start_time = time.time()

    # Process activations (filter dead components, concatenate)
    process_start = time.time()

    # Check if we received pre-filtered data from the interface
    if "filtered" in _activations_dict:
        # Interface passed pre-filtered data, create ProcessedActivations directly
        filtered_acts = _activations_dict["filtered"]

        # Use provided labels or create generic ones
        if "_labels" in _activations_dict:
            labels = _activations_dict["_labels"]
        else:
            n_components = filtered_acts.shape[1] if filtered_acts.ndim >= 2 else 0
            labels = [f"comp_{i}" for i in range(n_components)]

        # Get dead labels if provided
        dead_labels = _activations_dict.get("_dead_labels", [])

        # Create ProcessedActivations object directly
        from spd.clustering.activations import ProcessedActivations

        # Convert to numpy if it's a tensor
        acts_numpy = filtered_acts.numpy() if hasattr(filtered_acts, 'numpy') else filtered_acts

        processed_activations = ProcessedActivations(
            activations_raw={},  # No raw activations since we got pre-filtered data
            activations=acts_numpy,
            labels=labels,
            dead_components_lst=dead_labels
        )
    else:
        # Normal path with module structure
        processed_activations = process_activations(
            _activations_dict,
            filter_dead_threshold=config.filter_dead_threshold,
            seq_mode="concat",  # Concatenate sequence dimension for LM tasks
            filter_modules=None,  # No module filtering
            sort_components=False  # No component sorting
        )

    process_time = time.time() - process_start
    
    # Validate we have components to cluster
    if processed_activations.n_components_alive == 0:
        raise ValueError("No alive components found after filtering")
    
    # Convert to MergeConfig
    merge_config = config.to_merge_config()
    
    # Run clustering
    cluster_start = time.time()
    merge_history = merge_iteration(
        activations=processed_activations.activations,
        merge_config=merge_config,
        component_labels=processed_activations.labels,
        initial_merge=None,
        wandb_run=None,  # No WandB logging for comparison interface
        prefix="",
        plot_callback=None,
        artifact_callback=None
    )
    cluster_time = time.time() - cluster_start
    
    total_time = time.time() - start_time
    
    timing_info = {
        "total_time": total_time,
        "processing_time": process_time,
        "clustering_time": cluster_time,
        "num_iterations": merge_history.n_iters_current
    }
    
    return MDLClusteringResults(
        merge_history=merge_history,
        processed_activations=processed_activations,
        timing_info=timing_info,
        config=config
    )


def display_mdl_results(results: MDLClusteringResults) -> None:
    """Display MDL clustering results in Streamlit.
    
    Args:
        results: MDL clustering results
    """
    st.subheader("MDL Clustering Results")
    
    # Timing metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Time", f"{results.timing_info['total_time']:.2f}s")
    
    with col2:
        st.metric("Clustering Time", f"{results.timing_info['clustering_time']:.2f}s")
    
    with col3:
        st.metric("Iterations", results.timing_info['num_iterations'])
    
    with col4:
        st.metric("Final Groups", results.num_final_groups)
    
    # MDL cost (if available)
    if results.final_mdl_cost is not None:
        st.metric("Final MDL Cost (normalized)", f"{results.final_mdl_cost:.4f}")
    
    # Configuration used
    if st.checkbox("Show clustering configuration"):
        st.write("**Configuration:**")
        st.write(f"- Alpha: {results.config.alpha}")
        st.write(f"- Max iterations: {results.config.iters}")
        st.write(f"- Activation threshold: {results.config.activation_threshold}")
        st.write(f"- Dead component threshold: {results.config.filter_dead_threshold}")
        st.write(f"- Merge sampling: {results.config.merge_pair_sampling_method}")
    
    # Show final groups
    if st.checkbox("Show final component groups"):
        groups = results.final_groups
        st.write(f"**Final {len(groups)} groups:**")
        
        for group_id, components in groups.items():
            with st.expander(f"Group {group_id} ({len(components)} components)"):
                for component in components[:20]:  # Show first 20
                    st.write(f"- {component}")
                if len(components) > 20:
                    st.write(f"... and {len(components) - 20} more")


def get_clustering_summary(results: MDLClusteringResults) -> Dict[str, Any]:
    """Get a summary of clustering results for comparison.
    
    Args:
        results: MDL clustering results
        
    Returns:
        Dictionary with summary statistics
    """
    return {
        "method": "MDL Clustering",
        "total_time": results.timing_info['total_time'],
        "clustering_time": results.timing_info['clustering_time'],
        "num_iterations": results.timing_info['num_iterations'],
        "final_groups": results.num_final_groups,
        "final_cost": results.final_mdl_cost,
        "original_components": results.processed_activations.n_components_original,
        "alive_components": results.processed_activations.n_components_alive,
        "dead_components": results.processed_activations.n_components_dead,
    }