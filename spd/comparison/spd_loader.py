"""SPD Run Loader for clustering comparison interface."""

import streamlit as st
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from jaxtyping import Float, Int
from torch import Tensor

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import ModelPath


@st.cache_resource
def load_spd_run(run_path: str, device: str = "cpu") -> ComponentModel:
    """Load an SPD run and return the ComponentModel.
    
    Args:
        run_path: Path to SPD run (WandB path or local path)
        device: Device to load model on
        
    Returns:
        ComponentModel loaded from the run
    """
    spd_run_info = SPDRunInfo.from_path(run_path)
    component_model = ComponentModel.from_pretrained(spd_run_info.checkpoint_path)
    component_model.to(device)
    component_model.eval()
    return component_model


@st.cache_data
def get_component_activations(
    _component_model: ComponentModel, 
    batch_size: int = 1024,
    seq_len: int = 128,
    device: str = "cpu"
) -> Dict[str, Float[Tensor, "n_steps C"]]:
    """Get component activations from a ComponentModel using sample data.
    
    Args:
        _component_model: ComponentModel (underscore prefix for streamlit caching)
        batch_size: Size of batch to generate
        seq_len: Sequence length for generation
        device: Device to run on
        
    Returns:
        Dictionary of component activations per module
    """
    with torch.no_grad():
        # Generate random token IDs as sample data
        # This matches the expected input format for language models
        batch = torch.randint(
            0, 1000,  # Vocab size assumption
            (batch_size, seq_len),
            dtype=torch.long,
            device=device
        )
        
        # Get component activations
        _, pre_weight_acts = _component_model._forward_with_pre_forward_cache_hooks(
            batch, module_names=_component_model.target_module_paths
        )
        
        causal_importances, _ = _component_model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type="hard",  # Use hard sigmoid for binary activations
            detach_inputs=False,
        )
        
        return causal_importances


def validate_spd_run_path(run_path: str) -> bool:
    """Validate that an SPD run path is accessible.
    
    Args:
        run_path: Path to validate
        
    Returns:
        True if path is valid and accessible
    """
    try:
        SPDRunInfo.from_path(run_path)
        return True
    except Exception:
        return False


def get_run_info(run_path: str) -> Optional[Dict[str, Any]]:
    """Get basic information about an SPD run.
    
    Args:
        run_path: Path to SPD run
        
    Returns:
        Dictionary with run information or None if error
    """
    try:
        spd_run_info = SPDRunInfo.from_path(run_path)
        config = spd_run_info.config
        
        return {
            "experiment_type": getattr(config.task_config, 'task_name', 'unknown'),
            "num_components": config.C,
            "gate_type": config.gate_type,
            "checkpoint_path": str(spd_run_info.checkpoint_path),
            "config": config
        }
    except Exception as e:
        st.error(f"Error loading run info: {e}")
        return None