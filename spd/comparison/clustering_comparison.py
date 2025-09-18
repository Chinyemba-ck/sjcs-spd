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

                    # Try to get experiment type from various fields
                    if 'experiment' in run_config:
                        metadata['experiment'] = run_config['experiment']
                    elif 'pretrained_model_class' in run_config:
                        # Extract experiment type from model class path
                        # e.g., "spd.experiments.resid_mlp.models.ResidMLP" -> "resid_mlp"
                        model_class = run_config['pretrained_model_class']
                        if 'experiments' in model_class:
                            parts = model_class.split('.')
                            exp_idx = parts.index('experiments')
                            if exp_idx + 1 < len(parts):
                                metadata['experiment'] = parts[exp_idx + 1]

                    if 'model_id' in run_config:
                        metadata['model_id'] = run_config['model_id']
                    elif 'pretrained_model_id' in run_config:
                        metadata['model_id'] = run_config['pretrained_model_id']
            except Exception:
                pass

            # Find model files - be flexible with naming
            model_files = list(files_dir.glob("*.pth"))
            valid_model_found = False
            if model_files:
                # Accept any .pth file as potential model
                metadata['model_file'] = model_files[0].name
                metadata['all_model_files'] = [f.name for f in model_files]
                valid_model_found = True

            # Only add if we found a model file
            if valid_model_found:
                label = f"{metadata.get('experiment', 'Unknown')} ({run_dir.name[:8]})"
                runs.append((str(files_dir), label, metadata))

    return sorted(runs, key=lambda x: x[1])


@st.cache_data(ttl=300)  # Cache for 5 minutes
def discover_wandb_runs(project: str = "SJCS-SPD/spd", limit: int = 1000) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Discover SPD runs from W&B API.

    Args:
        project: W&B project path
        limit: Maximum number of runs to check (default 1000 to ensure we get all runs)

    Returns:
        List of (wandb_path, label, metadata) tuples
    """
    runs = []
    debug_info = {"total_checked": 0, "with_models": 0, "with_config": 0, "accepted": 0, "states_seen": {}}

    try:
        import wandb
        api = wandb.Api(timeout=30)

        # Get ALL runs from project - convert to list to ensure we fetch all pages
        # The api.runs() method returns a generator that fetches pages lazily
        # We don't filter by state to see everything
        st.caption(f"🔄 Fetching runs from {project}...")

        # Use a progress bar if possible
        progress_placeholder = st.empty()

        wandb_runs = api.runs(
            project,
            per_page=100  # Request more runs per page (max is usually 100)
        )

        # Debug: collect info about runs to understand filtering
        rejected_runs = []

        # Convert generator to list to ensure we get ALL runs (not just first page)
        all_runs = []
        for i, run in enumerate(wandb_runs):
            if i >= limit:
                break
            all_runs.append(run)
            # Update progress every 10 runs
            if i % 10 == 0:
                progress_placeholder.caption(f"📊 Fetched {i+1} runs so far...")

        progress_placeholder.caption(f"✅ Fetched {len(all_runs)} total runs from {project}")

        for i, run in enumerate(all_runs):
            debug_info["total_checked"] += 1

            # Track run states
            run_state = run.state if hasattr(run, 'state') else 'unknown'
            debug_info["states_seen"][run_state] = debug_info["states_seen"].get(run_state, 0) + 1

            # Check if it's an SPD run (has ComponentModel in files)
            files = [f.name for f in run.files()]

            # Check for ANY .pth files (SPD runs need at least one model file)
            all_pth_files = [f for f in files if f.endswith(".pth")]

            # SPD runs should have model_XXXXX.pth files specifically
            model_numbered_files = [f for f in all_pth_files if f.startswith("model_") and f.replace("model_", "").replace(".pth", "").isdigit()]

            # Accept runs with any .pth file (be inclusive)
            has_model_file = len(all_pth_files) > 0
            if has_model_file:
                debug_info["with_models"] += 1

            # The key differentiator: SPD runs have final_config.yaml
            has_spd_config = "final_config.yaml" in files
            if has_spd_config:
                debug_info["with_config"] += 1

            # Track why runs are rejected
            if not has_spd_config and not has_model_file:
                rejected_runs.append(f"{run.id[:8]} - No final_config.yaml or .pth files")
            elif not has_spd_config:
                rejected_runs.append(f"{run.id[:8]} - No final_config.yaml (has {len(all_pth_files)} .pth files)")
            elif not has_model_file:
                rejected_runs.append(f"{run.id[:8]} - No .pth files (has final_config.yaml)")

            # Accept runs with final_config.yaml and any .pth file
            if has_spd_config and has_model_file:
                debug_info["accepted"] += 1
                metadata = {
                    "run_id": run.id,
                    "type": "SPD",
                    "state": run.state,
                    "model_files": all_pth_files[:3],  # Store first 3 .pth files
                    "has_numbered_model": len(model_numbered_files) > 0,
                    "numbered_model": model_numbered_files[0] if model_numbered_files else None
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
                elif 'pretrained_model_class' in run.config:
                    # Extract from model class path
                    model_class = run.config['pretrained_model_class']
                    if 'experiments' in model_class:
                        parts = model_class.split('.')
                        exp_idx = parts.index('experiments')
                        if exp_idx + 1 < len(parts):
                            metadata['experiment'] = parts[exp_idx + 1]

                # Create label
                exp_name = metadata.get('experiment', 'Unknown')
                date_str = metadata.get('created_at', 'N/A')
                label = f"{exp_name} ({run.id[:8]}) - {date_str}"

                wandb_path = f"wandb:{project}/runs/{run.id}"
                runs.append((wandb_path, label, metadata))

        # ALWAYS show statistics for transparency and clear evidence
        st.info(f"✅ **W&B Run Discovery Complete**")
        states_summary = ", ".join([f"{state}: {count}" for state, count in debug_info["states_seen"].items()])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Runs Checked", debug_info['total_checked'])
        with col2:
            st.metric("SPD Runs Found", debug_info['accepted'])
        with col3:
            st.metric("Runs Filtered Out", debug_info['total_checked'] - debug_info['accepted'])

        # Show detailed breakdown
        st.caption(f"📊 Run states: {states_summary}")
        st.caption(f"📁 Files analysis: {debug_info['with_models']} had .pth files, {debug_info['with_config']} had final_config.yaml")

        # Show why runs were rejected if user wants details
        if rejected_runs:
            with st.expander(f"See why {len(rejected_runs)} runs were filtered out"):
                for rejection in rejected_runs[:30]:  # Show first 30
                    st.text(rejection)
                if len(rejected_runs) > 30:
                    st.text(f"... and {len(rejected_runs) - 30} more")

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
            # Refresh button with timestamp
            col1, col2 = st.columns([2, 1])
            with col1:
                # Show last update time if available
                from datetime import datetime
                if "last_refresh" not in st.session_state:
                    st.session_state.last_refresh = datetime.now()

                time_diff = (datetime.now() - st.session_state.last_refresh).total_seconds()
                if time_diff < 60:
                    time_str = f"{int(time_diff)}s ago"
                else:
                    time_str = f"{int(time_diff / 60)}m ago"
                st.caption(f"📅 Last refresh: {time_str}")

            with col2:
                if st.button("🔄 Force Refresh", help="Clear cache and reload all runs"):
                    st.cache_data.clear()
                    st.session_state.last_refresh = datetime.now()
                    st.rerun()

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
                project_options = {
                    "SJCS-SPD/spd-tms": "spd-tms (TMS decomposition runs)",
                    "SJCS-SPD/spd-resid-mlp": "spd-resid-mlp (ResidMLP decomposition runs)",
                    "SJCS-SPD/spd-lm": "spd-lm (Language model decomposition runs)",
                    "SJCS-SPD/spd": "spd (Mixed runs)",
                    "SJCS-SPD/spd-cluster": "spd-cluster (Clustering experiments)",
                    "SJCS-SPD/spd-train-tms": "spd-train-tms (TMS training runs)",
                    "SJCS-SPD/spd-train-resid-mlp": "spd-train-resid-mlp (ResidMLP training runs)",
                    "SJCS-SPD/seane-spd-tms": "seane-spd-tms (Sean's TMS experiments)"
                }

                wandb_project = st.selectbox(
                    "W&B Project:",
                    options=list(project_options.keys()),
                    format_func=lambda x: project_options[x],
                    index=0,  # Default to spd-tms
                    help="Select a project to browse SPD runs from"
                )
                if wandb_project:
                    with st.spinner(f"🔍 Fetching ALL runs from {wandb_project}... This may take a moment for large projects."):
                        wandb_runs = discover_wandb_runs(wandb_project)
                        discovered_runs.extend(wandb_runs)
                        if not wandb_runs and browse_source == "W&B Remote Runs":
                            st.warning(f"⚠️ No SPD runs found in {wandb_project}. Runs must have final_config.yaml and model files.")
                        elif wandb_runs:
                            st.success(f"✅ Found {len(wandb_runs)} SPD runs in {wandb_project}")

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
                        # Show compatibility note for model files
                        if 'model_files' in metadata:
                            st.caption("Model files: " + ", ".join(metadata['model_files']))
                            if metadata.get('has_numbered_model'):
                                st.info(f"Has standard model file: {metadata.get('numbered_model')}")
                            else:
                                st.warning("Non-standard model naming - may require special handling")
                        elif 'all_model_files' in metadata:
                            st.caption("Model files: " + ", ".join(metadata['all_model_files']))
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
                    
                    filtered_acts, filtered_labels, stats, dead_labels = filter_dead_components_with_stats(
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
                        # Pass both filtered activations and labels
                        filtered_dict = {
                            "filtered": filtered_acts,
                            "_labels": filtered_labels,  # Pass labels with underscore prefix
                            "_dead_labels": dead_labels  # Pass dead labels for ProcessedActivations
                        }

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