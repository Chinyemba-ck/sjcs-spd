"""
SPD Run Clustering Comparison Interface
Compare clustering methods side-by-side on SPD runs.
"""

import streamlit as st
import yaml
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import os

# Import our modules - add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from spd.comparison.spd_loader import load_spd_run, get_component_activations
from spd.comparison.dead_filter import filter_dead_components_with_stats
from spd.comparison.mdl_clustering import run_mdl_clustering, display_mdl_results, get_clustering_summary
from spd.comparison.metrics import MetricsTracker, estimate_mdl_flops
from spd.comparison.notebook_clustering import run_notebook_clustering, NotebookClusteringResults

# Page config
st.set_page_config(
    page_title="SPD Clustering Comparison",
    page_icon="🔬",
    layout="wide"
)

# Load config
config_path = Path(__file__).parent / "config.yaml"
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    st.error(f"Config file not found: {config_path}")
    st.stop()


@st.cache_data(ttl=60)  # Cache for 60 seconds to allow refresh
def discover_local_runs() -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Discover local SPD runs in wandb/ directory.

    Returns:
        List of (path, label, metadata) tuples
    """
    runs = []
    wandb_dir = Path("wandb")

    if not wandb_dir.exists():
        return runs

    for run_dir in wandb_dir.iterdir():
        if not run_dir.is_dir():
            continue

        files_dir = run_dir / "files"
        if not files_dir.exists():
            continue

        # Check if this is an SPD run (has final_config.yaml)
        config_path = files_dir / "final_config.yaml"
        if config_path.exists():
            # This is an SPD run
            metadata = {"run_id": run_dir.name, "type": "SPD"}

            # Try to read config for more info
            try:
                with open(config_path, 'r') as f:
                    run_config = yaml.safe_load(f)
                    if 'experiment' in run_config:
                        metadata['experiment'] = run_config['experiment']
                    if 'model_id' in run_config:
                        metadata['model_id'] = run_config['model_id']
            except Exception:
                pass

            # Find model files
            model_files = list(files_dir.glob("model_*.pth"))
            if model_files:
                metadata['model_file'] = model_files[0].name

            label = f"{metadata.get('experiment', 'Unknown')} ({run_dir.name[:8]})"
            runs.append((str(files_dir), label, metadata))

    return sorted(runs, key=lambda x: x[1])


@st.cache_data(ttl=300)  # Cache for 5 minutes
def discover_wandb_runs(project: str = "SJCS-SPD/spd", limit: int = 50) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Discover SPD runs from W&B API.

    Args:
        project: W&B project path
        limit: Maximum number of runs to fetch

    Returns:
        List of (wandb_path, label, metadata) tuples
    """
    runs = []

    try:
        import wandb
        api = wandb.Api(timeout=30)

        # Get runs from project
        wandb_runs = api.runs(project, filters={"state": "finished"})

        for i, run in enumerate(wandb_runs):
            if i >= limit:
                break

            # Check if it's an SPD run (has ComponentModel in files)
            files = [f.name for f in run.files()]
            has_model = any("model_" in f and f.endswith(".pth") for f in files)
            has_config = "final_config.yaml" in files

            if has_model and has_config:
                metadata = {
                    "run_id": run.id,
                    "type": "SPD",
                    "state": run.state,
                }

                # Safely get optional attributes
                if hasattr(run, 'created_at'):
                    metadata['created_at'] = run.created_at
                if hasattr(run, 'summary') and run.summary:
                    if '_runtime' in run.summary:
                        metadata['runtime_seconds'] = run.summary['_runtime']

                # Get experiment type from config
                if 'experiment' in run.config:
                    metadata['experiment'] = run.config['experiment']

                # Create label
                exp_name = metadata.get('experiment', 'Unknown')
                date_str = metadata.get('created_at', 'N/A')
                label = f"{exp_name} ({run.id[:8]}) - {date_str}"

                wandb_path = f"wandb:{project}/runs/{run.id}"
                runs.append((wandb_path, label, metadata))

    except Exception as e:
        st.warning(f"Could not fetch W&B runs: {e}")

    return runs


def display_notebook_results(results: NotebookClusteringResults):
    """Display notebook clustering results in a structured format."""
    st.subheader("Clustering Results")
    
    # Basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Components", results.total_components)
    with col2:
        st.metric("Clusters Found", results.n_clusters)
    with col3:
        st.metric("Similarity Method", results.similarity_method)
    
    # Cluster size distribution
    st.subheader("Cluster Sizes")
    cluster_sizes_data = []
    for cluster_id, size in results.cluster_sizes.items():
        cluster_sizes_data.append({"Cluster": f"Cluster {cluster_id}", "Size": size})
    
    if cluster_sizes_data:
        st.bar_chart({item["Cluster"]: item["Size"] for item in cluster_sizes_data})
    
    # Detailed cluster information
    with st.expander("Cluster Details"):
        for cluster_id in sorted(results.cluster_sizes.keys()):
            st.write(f"**Cluster {cluster_id}** ({results.cluster_sizes[cluster_id]} components)")
            members = results.get_cluster_members(cluster_id)
            
            # Show first few components in each cluster
            component_info = []
            for idx in members[:5]:  # Show first 5 components
                comp = results.components[idx]
                component_info.append(f"  - {comp.layer}:{comp.component_index}")
            
            st.text("\n".join(component_info))
            if len(members) > 5:
                st.text(f"  ... and {len(members) - 5} more")
            st.write("")


def main():
    st.title("SPD Run Clustering Comparison")
    st.markdown("Compare clustering methods on SPD runs (ResidMLP focus)")
    
    # Sidebar for input
    with st.sidebar:
        st.header("Configuration")
        
        # SPD run input
        st.subheader("SPD Run Selection")

        # Selection method
        selection_method = st.radio(
            "Selection method:",
            ["Browse Runs", "Manual Input"]
        )

        run_path = None

        if selection_method == "Browse Runs":
            # Refresh button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Refresh"):
                    st.cache_data.clear()

            # Run source for browsing
            browse_source = st.selectbox(
                "Browse from:",
                ["Local Cached Runs", "W&B Remote Runs", "Both"]
            )

            # Discover runs based on source
            discovered_runs = []

            if browse_source in ["Local Cached Runs", "Both"]:
                local_runs = discover_local_runs()
                discovered_runs.extend(local_runs)

            if browse_source in ["W&B Remote Runs", "Both"]:
                # W&B project selection
                wandb_project = st.text_input(
                    "W&B Project (optional):",
                    value="SJCS-SPD/spd",
                    help="Leave default or enter your W&B project"
                )
                if wandb_project:
                    wandb_runs = discover_wandb_runs(wandb_project)
                    discovered_runs.extend(wandb_runs)

            # Display discovered runs
            if discovered_runs:
                run_labels = ["Select a run..."] + [label for _, label, _ in discovered_runs]
                selected_index = st.selectbox(
                    "Available SPD runs:",
                    range(len(run_labels)),
                    format_func=lambda x: run_labels[x]
                )

                if selected_index > 0:
                    run_path, label, metadata = discovered_runs[selected_index - 1]
                    # Show metadata
                    with st.expander("Run Details"):
                        st.json(metadata)
            else:
                st.info("No SPD runs found. Check your wandb/ directory or W&B project.")

        else:  # Manual Input
            input_type = st.radio(
                "Input type:",
                ["Local Path", "W&B Path"]
            )

            if input_type == "W&B Path":
                run_path = st.text_input(
                    "W&B Run Path:",
                    help="Format: wandb:project/spd/runs/run_id - Must be an SPD run with ComponentModel"
                )
            else:
                run_path = st.text_input(
                    "Local Path:",
                    help="Path to local SPD run directory with final_config.yaml"
                )
        
        # Clustering parameters
        st.subheader("Clustering Parameters")
        
        dead_threshold = st.slider(
            "Dead Component Threshold",
            min_value=0.0,
            max_value=0.1,
            value=config.get("filter_dead_threshold", 0.01),
            step=0.001,
            format="%.3f",
            help="Components with max activation below this are filtered"
        )
        
        batch_size = st.number_input(
            "Batch size for activations",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100,
            help="Batch size for generating component activations"
        )
        
        device = st.selectbox(
            "Device",
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
            index=0
        )
        
        # MDL specific parameters
        st.subheader("MDL Parameters")
        mdl_alpha = st.slider(
            "Alpha (complexity penalty)",
            min_value=0.1,
            max_value=10.0,
            value=float(config["mdl_config"]["alpha"]),
            step=0.1
        )
        
        mdl_iters = st.slider(
            "Max iterations",
            min_value=10,
            max_value=500,
            value=config["mdl_config"]["iters"],
            step=10
        )
        
        # Notebook clustering specific parameters
        st.subheader("Notebook Clustering Parameters")
        similarity_method = st.selectbox(
            "Similarity Method",
            ["fused", "directional", "correlation", "coactivation"],
            index=0,
            help="Method for computing component similarity"
        )
        
        n_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=50,
            value=10,
            step=1,
            help="Number of clusters for SpectralClustering (None for auto-detect)"
        )
        
        n_samples = st.number_input(
            "Samples for gate computation",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of samples for computing gate profiles"
        )
        
        run_clustering = st.button("Run Clustering", type="primary")
    
    # Main content area - two columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.header("Notebook Clustering (Spectral)")
        
        if run_clustering and run_path:
            try:
                # Step 1: Load model (shared with MDL clustering)
                with st.spinner("Loading SPD model..."):
                    model = load_spd_run(run_path, device)
                    st.success(f"✅ Loaded model from {run_path}")
                
                # Step 2: Run notebook clustering
                with st.spinner("Running notebook clustering..."):
                    # Track metrics
                    tracker = MetricsTracker()
                    with tracker.track():
                        results = run_notebook_clustering(
                            component_model=model,
                            similarity_method=similarity_method,
                            n_clusters=n_clusters,
                            n_samples=n_samples,
                            random_state=42,  # For reproducibility
                            verbose=False
                        )
                    
                    metrics = tracker.get_metrics()
                
                # Step 3: Display results
                st.success("✅ Notebook Clustering complete!")
                
                # Display metrics
                st.subheader("Performance Metrics")
                metrics_dict = metrics.to_dict()
                col1, col2 = st.columns(2)
                with col1:
                    for key in list(metrics_dict.keys())[:3]:
                        st.metric(key, metrics_dict[key])
                with col2:
                    for key in list(metrics_dict.keys())[3:]:
                        st.metric(key, metrics_dict[key])
                
                # Display clustering results
                display_notebook_results(results)
                
                # Store results in session state for comparison
                if "clustering_results" not in st.session_state:
                    st.session_state.clustering_results = {}
                st.session_state.clustering_results["notebook"] = {
                    "results": results,
                    "metrics": metrics,
                    "summary": {
                        "method": f"Spectral ({similarity_method})",
                        "total_time": metrics.wall_time,
                        "final_groups": results.n_clusters,
                        "total_components": results.total_components
                    }
                }
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
        
        elif run_clustering:
            st.warning("Please provide an SPD run path")
    
    with col_right:
        st.header("MDL Clustering")
        
        if run_clustering and run_path:
            try:
                # Step 1: Load model
                with st.spinner("Loading SPD model..."):
                    model = load_spd_run(run_path, device)
                    st.success(f"✅ Loaded model from {run_path}")
                
                # Step 2: Get activations
                with st.spinner("Generating component activations..."):
                    activations_dict = get_component_activations(
                        model, 
                        batch_size=batch_size,
                        device=device
                    )
                    
                    # Show activation stats
                    total_components = sum(act.shape[-1] for act in activations_dict.values())
                    st.write(f"Generated activations for {total_components} components across {len(activations_dict)} modules")
                
                # Step 3: Filter dead components
                with st.spinner("Filtering dead components..."):
                    # Concatenate all activations for dead filtering
                    all_acts = torch.cat(list(activations_dict.values()), dim=-1)
                    labels = []
                    for module_name, acts in activations_dict.items():
                        for i in range(acts.shape[-1]):
                            labels.append(f"{module_name}:{i}")
                    
                    filtered_acts, filtered_labels, stats = filter_dead_components_with_stats(
                        all_acts, labels, dead_threshold
                    )
                    
                    # Display filtering stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original", stats["original"])
                    with col2:
                        st.metric("Alive", stats["alive"])
                    with col3:
                        st.metric("Dead", stats["dead"])
                
                # Step 4: Run MDL clustering
                with st.spinner("Running MDL clustering..."):
                    # Update config with user parameters
                    mdl_config = config["mdl_config"].copy()
                    mdl_config["alpha"] = mdl_alpha
                    mdl_config["iters"] = mdl_iters
                    
                    # Track metrics
                    tracker = MetricsTracker()
                    with tracker.track():
                        # Need to reconstruct activations_dict for MDL clustering
                        # This is a simplified version - in production you'd maintain the structure
                        filtered_dict = {"filtered": filtered_acts}
                        
                        results = run_mdl_clustering(
                            filtered_dict,
                            mdl_config
                        )
                    
                    metrics = tracker.get_metrics()
                    
                    # Add theoretical FLOPs
                    metrics.theoretical_flops = estimate_mdl_flops(
                        n_components=stats["alive"],
                        n_samples=batch_size,
                        n_iterations=results.timing_info['num_iterations']
                    )
                
                # Step 5: Display results
                st.success("✅ MDL Clustering complete!")
                
                # Display metrics
                st.subheader("Performance Metrics")
                metrics_dict = metrics.to_dict()
                col1, col2 = st.columns(2)
                with col1:
                    for key in list(metrics_dict.keys())[:3]:
                        st.metric(key, metrics_dict[key])
                with col2:
                    for key in list(metrics_dict.keys())[3:]:
                        st.metric(key, metrics_dict[key])
                
                # Display clustering results
                display_mdl_results(results)
                
                # Store results in session state for comparison
                if "clustering_results" not in st.session_state:
                    st.session_state.clustering_results = {}
                st.session_state.clustering_results["mdl"] = {
                    "results": results,
                    "metrics": metrics,
                    "summary": get_clustering_summary(results)
                }
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
        
        elif run_clustering:
            st.warning("Please provide an SPD run path")
    
    # Comparison section (at bottom)
    if "clustering_results" in st.session_state and len(st.session_state.clustering_results) > 0:
        st.divider()
        st.header("Results Summary")
        
        # Create comparison table for both methods
        results = st.session_state.clustering_results
        
        if len(results) == 2:  # Both methods completed
            st.subheader("Side-by-Side Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                if "notebook" in results:
                    nb_summary = results["notebook"]["summary"]
                    st.write("**Notebook Clustering**")
                    st.metric("Method", nb_summary["method"])
                    st.metric("Time", f"{nb_summary['total_time']:.2f}s")
                    st.metric("Clusters", nb_summary["final_groups"])
                    st.metric("Components", nb_summary["total_components"])
            
            with col2:
                if "mdl" in results:
                    mdl_summary = results["mdl"]["summary"]
                    st.write("**MDL Clustering**")
                    st.metric("Method", mdl_summary["method"])
                    st.metric("Time", f"{mdl_summary['total_time']:.2f}s")
                    st.metric("Clusters", mdl_summary["final_groups"])
                    st.metric("Final Cost", f"{mdl_summary['final_cost']:.4f}")
        
        else:  # Only one method completed
            if "notebook" in results:
                summary = results["notebook"]["summary"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Method", summary["method"])
                with col2:
                    st.metric("Total Time", f"{summary['total_time']:.2f}s")
                with col3:
                    st.metric("Final Groups", summary["final_groups"])
                with col4:
                    st.metric("Total Components", summary["total_components"])
            
            elif "mdl" in results:
                summary = results["mdl"]["summary"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Method", summary["method"])
                with col2:
                    st.metric("Total Time", f"{summary['total_time']:.2f}s")
                with col3:
                    st.metric("Final Groups", summary["final_groups"])
                with col4:
                    st.metric("Final MDL Cost", f"{summary['final_cost']:.4f}")


if __name__ == "__main__":
    main()