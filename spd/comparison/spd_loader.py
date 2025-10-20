"""SPD Run Loader for clustering comparison interface."""

import streamlit as st
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from jaxtyping import Float, Int
from torch import Tensor

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import ModelPath


class ModelLoadError(Exception):
    """Custom exception for model loading errors with detailed information."""
    def __init__(self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


@st.cache_resource
def load_spd_run(run_path: str, device: str = "cpu") -> ComponentModel:
    """Load an SPD run and return the ComponentModel.

    Args:
        run_path: Path to SPD run (WandB path or local path)
        device: Device to load model on

    Returns:
        ComponentModel loaded from the run

    Raises:
        ModelLoadError: If the model cannot be loaded with detailed error information
    """
    # If it's a local wandb cache path, ensure we point to the files directory
    if not run_path.startswith("wandb:") and "wandb" in run_path:
        path = Path(run_path)
        # If path doesn't end with 'files', add it
        if path.name != "files" and (path / "files").exists():
            run_path = str(path / "files")

    # Try to load run info
    try:
        spd_run_info = SPDRunInfo.from_path(run_path)
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load run configuration from {run_path}",
            error_type="config_load_error",
            details={"path": run_path, "error": str(e)}
        )

    # Try to load the model
    try:
        component_model = ComponentModel.from_pretrained(spd_run_info.checkpoint_path)
    except RuntimeError as e:
        error_str = str(e)

        # Check for state dict key mismatches
        if "Missing key(s) in state_dict" in error_str and "Unexpected key(s)" in error_str:
            # Extract missing and unexpected keys
            missing_keys = []
            unexpected_keys = []

            if "Missing key(s)" in error_str:
                missing_start = error_str.find('Missing key(s) in state_dict: "') + len('Missing key(s) in state_dict: "')
                missing_end = error_str.find('"', missing_start)
                if missing_end > missing_start:
                    missing_keys = error_str[missing_start:missing_end].split('", "')

            if "Unexpected key(s)" in error_str:
                unexpected_start = error_str.find('Unexpected key(s) in state_dict: "') + len('Unexpected key(s) in state_dict: "')
                unexpected_end = error_str.find('"', unexpected_start)
                if unexpected_end > unexpected_start:
                    unexpected_keys = error_str[unexpected_start:unexpected_end].split('", "')

            # Determine if this is an architecture mismatch
            is_old_format = any(k.startswith("model.") for k in unexpected_keys)
            is_architecture_mismatch = len(missing_keys) > 5 or len(unexpected_keys) > 5

            if is_old_format:
                error_msg = "This checkpoint uses an older model format that is incompatible with the current code."
                error_type = "old_format_error"
            elif is_architecture_mismatch:
                error_msg = "The checkpoint has a different model architecture than expected (different layer names or sizes)."
                error_type = "architecture_mismatch"
            else:
                error_msg = "State dict keys don't match between checkpoint and model."
                error_type = "state_dict_mismatch"

            raise ModelLoadError(
                error_msg,
                error_type=error_type,
                details={
                    "missing_keys": missing_keys[:5],  # Show first 5
                    "unexpected_keys": unexpected_keys[:5],  # Show first 5
                    "total_missing": len(missing_keys),
                    "total_unexpected": len(unexpected_keys),
                    "checkpoint_path": str(spd_run_info.checkpoint_path)
                }
            )

        # Check for tensor shape mismatches
        elif "size mismatch" in error_str:
            # Extract shape information
            import re
            shape_pattern = r"shape torch\.Size\((\[[^\]]+\])\)"
            shapes = re.findall(shape_pattern, error_str)

            raise ModelLoadError(
                "The checkpoint has incompatible tensor shapes (likely due to different architecture parameters).",
                error_type="tensor_shape_mismatch",
                details={
                    "error": error_str[:500],  # First 500 chars of error
                    "checkpoint_path": str(spd_run_info.checkpoint_path),
                    "config": {
                        "gate_type": spd_run_info.config.gate_type if hasattr(spd_run_info.config, 'gate_type') else None,
                        "gate_hidden_dims": spd_run_info.config.gate_hidden_dims if hasattr(spd_run_info.config, 'gate_hidden_dims') else None,
                        "C": spd_run_info.config.C if hasattr(spd_run_info.config, 'C') else None
                    }
                }
            )
        else:
            # Generic runtime error
            raise ModelLoadError(
                f"Failed to load model checkpoint: {str(e)[:200]}",
                error_type="runtime_error",
                details={"checkpoint_path": str(spd_run_info.checkpoint_path), "error": str(e)}
            )

    except Exception as e:
        # Catch any other unexpected errors
        raise ModelLoadError(
            f"Unexpected error loading model: {str(e)[:200]}",
            error_type="unexpected_error",
            details={"checkpoint_path": str(spd_run_info.checkpoint_path), "error": str(e)}
        )

    component_model.to(device)
    component_model.eval()
    return component_model


@st.cache_data
def get_component_activations(
    _component_model: ComponentModel,
    batch_size: int = 1024,
    seq_len: int = 128,
    device: str = "cpu",
    sampling: Literal["continuous", "binomial"] = "continuous"
) -> Dict[str, Float[Tensor, "n_steps C"]]:
    """Get component activations from a ComponentModel using sample data.

    Args:
        _component_model: ComponentModel (underscore prefix for streamlit caching)
        batch_size: Size of batch to generate
        seq_len: Sequence length for generation (used for language models)
        device: Device to run on
        sampling: Sampling method for causal importance calculation ("continuous" or "binomial")

    Returns:
        Dictionary of component activations per module
    """
    with torch.no_grad():
        # Check model type to generate appropriate input
        model_name = _component_model.patched_model.__class__.__name__
        
        if "TMS" in model_name or "ToyModel" in model_name:
            # TMS models expect (batch_size, n_features) float input
            # Get input dimension from model
            try:
                # Try to get n_features from model config
                n_features = _component_model.patched_model.n_features
            except AttributeError:
                # Default to 100 if not found
                n_features = 100
            
            batch = torch.randn(batch_size, n_features, device=device)
        elif "ResidMLP" in model_name:
            # ResidMLP models expect (batch_size, n_features) float input
            try:
                n_features = _component_model.patched_model.n_features
            except AttributeError:
                n_features = 100
            
            batch = torch.randn(batch_size, n_features, device=device)
        else:
            # Language models expect (batch_size, seq_len) long input
            # Try to get vocab size from model config
            try:
                vocab_size = _component_model.patched_model.config.vocab_size
            except AttributeError:
                # Fallback to a safe large value that works for most models
                vocab_size = 50000  # Common for GPT-2 (50257), BERT (30522), etc.

            batch = torch.randint(
                0, vocab_size,
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
            sampling=sampling,
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