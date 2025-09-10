import os
from pathlib import Path
from typing import Dict, Any, Optional

import wandb
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from dotenv import load_dotenv

# Will use pandas from datasets package
try:
    import pandas as pd
except ImportError:
    # Pandas is available through datasets dependency
    from datasets.utils import pandas as pd

# Load environment variables
load_dotenv()


def fetch_run_data(run_path: str) -> Optional[Dict[str, Any]]:
    """
    Fetch all data from a W&B run.
    
    Args:
        run_path: Path to W&B run in format 'entity/project/run_id'
                  e.g., 'goodfire/spd/uz2lmdjz'
    
    Returns:
        Dict containing:
            - run: The W&B run object
            - summary: Final metric values (run.summary)
            - history: Training history dataframe (run.history())
            - config: Run configuration (run.config)
            - images: Dict of logged images
        Returns None if run not found or error occurs.
    """
    try:
        # Initialize W&B API
        api = wandb.Api(timeout=30)
        
        # Get the run
        run = api.run(run_path)
        
        # Extract data
        data = {
            "run": run,
            "summary": dict(run.summary) if run.summary else {},
            "history": run.history(),
            "config": dict(run.config) if run.config else {},
            "images": {}  # Will implement image fetching later
        }
        
        return data
        
    except Exception as e:
        print(f"Error fetching run {run_path}: {e}")
        return None


def get_run_metrics(run_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract key metrics from run summary into a pandas DataFrame.
    
    Args:
        run_data: Dictionary from fetch_run_data containing run information
    
    Returns:
        DataFrame with metric names and values
    """
    if not run_data or "summary" not in run_data:
        return pd.DataFrame()
    
    summary = run_data["summary"]
    
    # Filter out system metrics (those starting with underscore)
    metrics = {k: v for k, v in summary.items() if not k.startswith("_")}
    
    # Convert to DataFrame
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    
    # Sort by metric name for consistent display
    df = df.sort_values("Metric").reset_index(drop=True)
    
    return df


def get_run_history(run_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract training history from a W&B run.
    
    Args:
        run_data: Dictionary from fetch_run_data containing run information
    
    Returns:
        DataFrame with training history (steps, losses, metrics over time)
    """
    if not run_data or "history" not in run_data:
        return pd.DataFrame()
    
    history = run_data["history"]
    
    # History is already a DataFrame from W&B
    if history.empty:
        return pd.DataFrame()
    
    # Filter out system columns (those starting with underscore)
    user_columns = [col for col in history.columns if not col.startswith("_")]
    
    # Return filtered history
    return history[user_columns].copy()


def get_run_config(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract run configuration from a W&B run.
    
    Args:
        run_data: Dictionary from fetch_run_data containing run information
    
    Returns:
        Dictionary with configuration parameters
    """
    if not run_data or "config" not in run_data:
        return {}
    
    config = run_data["config"]
    
    # Return a copy of the config
    return dict(config)


def get_run_images(run_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract image URLs from a W&B run summary.
    
    Args:
        run_data: Dictionary from fetch_run_data containing run information
    
    Returns:
        Dictionary mapping image names to their W&B URLs
    """
    if not run_data or "summary" not in run_data:
        return {}
    
    summary = run_data["summary"]
    run = run_data.get("run")
    
    images = {}
    
    # Look for image-file entries in summary
    for key, value in summary.items():
        # Check if value has dict-like interface (handles SummarySubDict)
        if hasattr(value, 'get') and value.get("_type") == "image-file":
            # Build W&B media URL
            if run and "path" in value:
                # W&B media URL format
                entity = run.entity
                project = run.project
                run_id = run.id
                file_path = value["path"]
                
                # Construct the media URL
                url = f"https://api.wandb.ai/files/{entity}/{project}/{run_id}/{file_path}"
                images[key] = url
    
    return images