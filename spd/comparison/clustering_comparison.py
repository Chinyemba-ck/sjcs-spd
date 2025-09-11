"""
SPD Run Clustering Comparison Interface
Compare clustering methods side-by-side on SPD runs.
"""

import streamlit as st
import yaml
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# Import our modules - add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from spd.comparison.spd_loader import load_spd_run, get_component_activations
from spd.comparison.dead_filter import filter_dead_components_with_stats
from spd.comparison.mdl_clustering import run_mdl_clustering, display_mdl_results, get_clustering_summary
from spd.comparison.metrics import MetricsTracker, estimate_mdl_flops

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


def main():
    st.title("SPD Run Clustering Comparison")
    st.markdown("Compare clustering methods on SPD runs (ResidMLP focus)")
    
    # Sidebar for input
    with st.sidebar:
        st.header("Configuration")
        
        # SPD run input
        st.subheader("SPD Run Selection")
        run_source = st.radio(
            "Run source:",
            ["W&B Path", "Local Path"]
        )
        
        if run_source == "W&B Path":
            run_path = st.text_input(
                "W&B Run Path:",
                help="Format: wandb:project/spd/runs/run_id - Must be an SPD run with ComponentModel"
            )
        else:
            run_path = st.text_input(
                "Local Path:",
                help="Path to local SPD run directory with final_config.yaml or similar"
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
        
        run_clustering = st.button("Run Clustering", type="primary")
    
    # Main content area - two columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.header("Custom Clustering (Future)")
        st.info("This column is reserved for your custom clustering method.")
        st.markdown("""
        ### Placeholder for future implementation
        - Will use same preprocessed activations
        - Will display same metrics format
        - Ready for side-by-side comparison
        """)
    
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
        
        # Display comparison table
        if "mdl" in st.session_state.clustering_results:
            summary = st.session_state.clustering_results["mdl"]["summary"]
            
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