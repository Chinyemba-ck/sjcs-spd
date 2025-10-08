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


def plot_training_curves(run1_data: Dict[str, Any], run2_data: Dict[str, Any], metric_name: str) -> matplotlib.figure.Figure:
    """
    Plot training curves for a specific metric from both runs.
    
    Args:
        run1_data: Data from first run
        run2_data: Data from second run
        metric_name: Name of metric to plot
    
    Returns:
        Matplotlib figure
    """
    history1 = get_run_history(run1_data)
    history2 = get_run_history(run2_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data if available
    if not history1.empty and metric_name in history1.columns:
        steps1 = range(len(history1))
        ax.plot(steps1, history1[metric_name], label="Run 1", alpha=0.8)
    
    if not history2.empty and metric_name in history2.columns:
        steps2 = range(len(history2))
        ax.plot(steps2, history2[metric_name], label="Run 2", alpha=0.8)
    
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Training Curves: {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def display_figures(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get figure URLs for side-by-side comparison.
    
    Args:
        run1_data: Data from first run
        run2_data: Data from second run
    
    Returns:
        Dictionary with figure names and URLs for both runs
    """
    images1 = get_run_images(run1_data)
    images2 = get_run_images(run2_data)
    
    # Get all unique figure names
    all_figures = set(list(images1.keys()) + list(images2.keys()))
    
    figures = {}
    for fig_name in sorted(all_figures):
        figures[fig_name] = {
            'run1': images1.get(fig_name),
            'run2': images2.get(fig_name)
        }
    
    return figures


def compare_configs(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare configurations between two runs.
    
    Args:
        run1_data: Data from first run
        run2_data: Data from second run
    
    Returns:
        DataFrame with config comparison
    """
    config1 = get_run_config(run1_data)
    config2 = get_run_config(run2_data)
    
    # Get all unique config keys
    all_keys = set(list(config1.keys()) + list(config2.keys()))
    
    comparison_data = []
    for key in sorted(all_keys):
        val1 = config1.get(key, "N/A")
        val2 = config2.get(key, "N/A")
        
        # Truncate long values
        if isinstance(val1, str) and len(val1) > 50:
            val1 = val1[:50] + "..."
        if isinstance(val2, str) and len(val2) > 50:
            val2 = val2[:50] + "..."
        
        # Check if values are different
        is_different = str(val1) != str(val2)
        
        comparison_data.append({
            'Parameter': key,
            'Run 1': str(val1),
            'Run 2': str(val2),
            'Different': '✓' if is_different else ''
        })
    
    return pd.DataFrame(comparison_data)


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
        
        # Create tabs for different comparisons
        tabs = st.tabs(["📊 Metrics", "📈 Training Curves", "🖼️ Figures", "⚙️ Config"])
        
        # Metrics Tab
        with tabs[0]:
            st.header("Metrics Comparison")
            metrics_df = compare_metrics(run1_data, run2_data)
            if not metrics_df.empty:
                # Filter out non-numeric metrics for cleaner display
                numeric_df = metrics_df[metrics_df['Difference'] != '-'].copy()
                
                if not numeric_df.empty:
                    st.subheader("Numeric Metrics")
                    st.dataframe(numeric_df, use_container_width=True)
                
                # Show all metrics
                with st.expander("Show All Metrics"):
                    st.dataframe(metrics_df, use_container_width=True)
            else:
                st.warning("No metrics available for comparison.")
        
        # Training Curves Tab
        with tabs[1]:
            st.header("Training Curves")
            
            # Get available metrics from history
            history1 = get_run_history(run1_data)
            history2 = get_run_history(run2_data)
            
            # Get common metrics that exist in at least one run
            available_metrics = set()
            if not history1.empty:
                available_metrics.update(history1.columns)
            if not history2.empty:
                available_metrics.update(history2.columns)
            
            # Filter to numeric columns only
            numeric_metrics = []
            for metric in available_metrics:
                if not history1.empty and metric in history1.columns:
                    if pd.api.types.is_numeric_dtype(history1[metric]):
                        numeric_metrics.append(metric)
                elif not history2.empty and metric in history2.columns:
                    if pd.api.types.is_numeric_dtype(history2[metric]):
                        numeric_metrics.append(metric)
            
            if numeric_metrics:
                # Metric selector
                selected_metric = st.selectbox(
                    "Select metric to plot",
                    sorted(set(numeric_metrics)),
                    index=0 if numeric_metrics else None
                )
                
                if selected_metric:
                    fig = plot_training_curves(run1_data, run2_data, selected_metric)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("No numeric metrics available in training history.")
        
        # Figures Tab
        with tabs[2]:
            st.header("Figure Comparison")
            figures = display_figures(run1_data, run2_data)
            
            if figures:
                for fig_name, urls in figures.items():
                    st.subheader(fig_name.replace('_', ' ').title())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if urls['run1']:
                            st.image(urls['run1'], caption="Run 1", use_container_width=True)
                        else:
                            st.info("Not available in Run 1")
                    
                    with col2:
                        if urls['run2']:
                            st.image(urls['run2'], caption="Run 2", use_container_width=True)
                        else:
                            st.info("Not available in Run 2")
                    
                    st.divider()
            else:
                st.warning("No figures available for comparison.")
        
        # Config Tab
        with tabs[3]:
            st.header("Configuration Comparison")
            config_df = compare_configs(run1_data, run2_data)
            
            if not config_df.empty:
                # Show differences first
                diff_df = config_df[config_df['Different'] == '✓']
                if not diff_df.empty:
                    st.subheader("Different Parameters")
                    st.dataframe(diff_df, use_container_width=True)
                
                # Show all parameters
                with st.expander("Show All Parameters"):
                    st.dataframe(config_df, use_container_width=True)
            else:
                st.warning("No configuration available for comparison.")
    else:
        st.info("Please enter run paths in the sidebar and click 'Fetch Run Data' to begin.")


if __name__ == "__main__":
    main()