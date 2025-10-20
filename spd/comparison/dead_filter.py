"""Dead component filtering for clustering comparison interface."""

import streamlit as st
from typing import Dict, Tuple, List
from jaxtyping import Float, Bool
from torch import Tensor


def filter_dead_components(
    activations: Float[Tensor, "n_steps c"],
    labels: List[str],
    threshold: float = 0.01
) -> Tuple[Float[Tensor, "n_steps c_alive"], List[str], List[str]]:
    """Filter out dead components based on maximum activation threshold.
    
    Args:
        activations: Component activations tensor
        labels: Component labels
        threshold: Threshold for considering a component dead
        
    Returns:
        Tuple of (filtered_activations, alive_labels, dead_labels)
    """
    if threshold <= 0:
        return activations, labels, []
    
    # Find components with max activation below threshold
    max_activations = activations.max(dim=0).values
    alive_mask = max_activations >= threshold
    
    # Filter activations and labels
    filtered_activations = activations[:, alive_mask]
    alive_labels = [label for i, label in enumerate(labels) if alive_mask[i]]
    dead_labels = [label for i, label in enumerate(labels) if not alive_mask[i]]
    
    return filtered_activations, alive_labels, dead_labels


@st.cache_data
def apply_dead_filtering(
    activations_dict: Dict[str, Float[Tensor, "n_steps C"]], 
    threshold: float = 0.01
) -> Dict[str, any]:
    """Apply dead component filtering to all modules and return results.
    
    Args:
        activations_dict: Dictionary of activations per module
        threshold: Dead component threshold
        
    Returns:
        Dictionary containing filtered results and statistics
    """
    results = {
        "filtered_activations": {},
        "alive_labels": {},
        "dead_labels": {},
        "stats": {}
    }
    
    total_original = 0
    total_alive = 0
    total_dead = 0
    
    for module_name, activations in activations_dict.items():
        # Create labels for this module
        n_components = activations.shape[1]
        labels = [f"{module_name}:{i}" for i in range(n_components)]
        
        # Apply filtering
        filtered_acts, alive_labels, dead_labels = filter_dead_components(
            activations, labels, threshold
        )
        
        # Store results
        results["filtered_activations"][module_name] = filtered_acts
        results["alive_labels"][module_name] = alive_labels
        results["dead_labels"][module_name] = dead_labels
        
        # Update totals
        total_original += n_components
        total_alive += len(alive_labels)
        total_dead += len(dead_labels)
    
    # Overall statistics
    results["stats"] = {
        "total_original": total_original,
        "total_alive": total_alive,
        "total_dead": total_dead,
        "dead_percentage": (total_dead / total_original * 100) if total_original > 0 else 0,
        "threshold_used": threshold
    }
    
    return results


def filter_dead_components_with_stats(
    activations: Float[Tensor, "n_steps c"],
    labels: List[str],
    threshold: float = 0.01
) -> Tuple[Float[Tensor, "n_steps c_alive"], List[str], Dict[str, int], List[str]]:
    """Filter dead components and return statistics.

    Args:
        activations: Component activations tensor
        labels: Component labels
        threshold: Threshold for considering a component dead

    Returns:
        Tuple of (filtered_activations, alive_labels, stats_dict, dead_labels)
    """
    filtered_acts, alive_labels, dead_labels = filter_dead_components(
        activations, labels, threshold
    )

    stats = {
        "original": len(labels),
        "alive": len(alive_labels),
        "dead": len(dead_labels)
    }

    return filtered_acts, alive_labels, stats, dead_labels


def display_filtering_stats(filter_results: Dict[str, any]) -> None:
    """Display filtering statistics in Streamlit.
    
    Args:
        filter_results: Results from apply_dead_filtering
    """
    stats = filter_results["stats"]
    
    st.subheader("Dead Component Filtering Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Components", stats["total_original"])
    
    with col2:
        st.metric("Alive Components", stats["total_alive"])
    
    with col3:
        st.metric("Dead Components", stats["total_dead"])
    
    with col4:
        st.metric("Dead %", f"{stats['dead_percentage']:.1f}%")
    
    st.write(f"**Threshold:** {stats['threshold_used']}")
    
    # Show per-module breakdown if requested
    if st.checkbox("Show per-module breakdown"):
        for module_name in filter_results["alive_labels"].keys():
            n_alive = len(filter_results["alive_labels"][module_name])
            n_dead = len(filter_results["dead_labels"][module_name])
            total = n_alive + n_dead
            dead_pct = (n_dead / total * 100) if total > 0 else 0
            
            st.write(f"**{module_name}:** {n_alive} alive, {n_dead} dead ({dead_pct:.1f}%)")