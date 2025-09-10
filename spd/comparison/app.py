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


def compare_metrics(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare metrics between two runs.
    
    Args:
        run1_data: Data from first run
        run2_data: Data from second run
    
    Returns:
        DataFrame with side-by-side comparison
    """
    metrics1 = get_run_metrics(run1_data)
    metrics2 = get_run_metrics(run2_data)
    
    # Get all unique metric names
    all_metrics = set()
    if not metrics1.empty:
        all_metrics.update(metrics1['Metric'].values)
    if not metrics2.empty:
        all_metrics.update(metrics2['Metric'].values)
    
    # Build comparison dataframe
    comparison_data = []
    for metric in sorted(all_metrics):
        row = {'Metric': metric}
        
        # Get values from each run
        value1 = metrics1[metrics1['Metric'] == metric]['Value'].values
        value2 = metrics2[metrics2['Metric'] == metric]['Value'].values
        
        # Handle numeric values
        if value1.size > 0:
            val1 = value1[0]
            if isinstance(val1, (int, float)):
                row['Run 1'] = f"{val1:.6g}"
            else:
                row['Run 1'] = str(val1)[:50]  # Truncate long strings
        else:
            row['Run 1'] = "N/A"
            
        if value2.size > 0:
            val2 = value2[0]
            if isinstance(val2, (int, float)):
                row['Run 2'] = f"{val2:.6g}"
            else:
                row['Run 2'] = str(val2)[:50]  # Truncate long strings
        else:
            row['Run 2'] = "N/A"
        
        # Calculate difference for numeric values
        if value1.size > 0 and value2.size > 0:
            if isinstance(value1[0], (int, float)) and isinstance(value2[0], (int, float)):
                diff = value2[0] - value1[0]
                if value1[0] != 0:
                    pct_diff = (diff / abs(value1[0])) * 100
                    row['Difference'] = f"{diff:.6g} ({pct_diff:+.1f}%)"
                else:
                    row['Difference'] = f"{diff:.6g}"
            else:
                row['Difference'] = "-"
        else:
            row['Difference'] = "-"
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def main():
    """Main Streamlit app for comparing SPD runs."""
    st.set_page_config(
        page_title="SPD Run Comparison",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("SPD Run Comparison Dashboard")
    
    # Sidebar for run selection
    with st.sidebar:
        st.header("Select Runs to Compare")
        
        # Input fields for run paths
        run1_path = st.text_input(
            "Run 1 Path",
            placeholder="entity/project/run_id",
            help="Format: entity/project/run_id (e.g., goodfire/spd/uz2lmdjz)"
        )
        
        run2_path = st.text_input(
            "Run 2 Path", 
            placeholder="entity/project/run_id",
            help="Format: entity/project/run_id (e.g., goodfire/spd/w6zdzkfp)"
        )
        
        # Fetch button
        fetch_button = st.button("Fetch Run Data", type="primary")
        
        # Status placeholder
        status_placeholder = st.empty()
    
    # Main content area
    if fetch_button and run1_path and run2_path:
        with status_placeholder:
            st.info("Fetching run data...")
        
        # Fetch data for both runs
        run1_data = fetch_run_data(run1_path)
        run2_data = fetch_run_data(run2_path)
        
        if run1_data and run2_data:
            with status_placeholder:
                st.success("Successfully fetched both runs!")
            
            # Store in session state
            st.session_state['run1_data'] = run1_data
            st.session_state['run2_data'] = run2_data
            st.session_state['run1_path'] = run1_path
            st.session_state['run2_path'] = run2_path
        else:
            with status_placeholder:
                st.error("Failed to fetch one or both runs. Please check the run paths.")
    
    # Display comparison if data is available
    if 'run1_data' in st.session_state and 'run2_data' in st.session_state:
        run1_data = st.session_state['run1_data']
        run2_data = st.session_state['run2_data']
        run1_path = st.session_state['run1_path']
        run2_path = st.session_state['run2_path']
        
        # Display run names
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Run 1: {run1_path.split('/')[-1]}")
        with col2:
            st.subheader(f"Run 2: {run2_path.split('/')[-1]}")
        
        # Placeholder for tabs (will be implemented next)
        st.info("Comparison features will be added here...")
    else:
        st.info("Please enter run paths in the sidebar and click 'Fetch Run Data' to begin.")


if __name__ == "__main__":
    main()