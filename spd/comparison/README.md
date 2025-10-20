# SPD Comparison Tools

This directory contains two Streamlit applications for analyzing SPD (Sparse Parameter Decomposition) runs:

1. **SPD Run Comparison Dashboard** (`app.py`) - Compare metrics and visualizations between W&B runs
2. **Clustering Comparison Interface** (`clustering_comparison.py`) - Compare clustering methods on SPD models

---

## 1. SPD Run Comparison Dashboard (`app.py`)

### Features

- **Metrics Comparison**: Side-by-side final metrics with percentage differences
- **Training Curves**: Overlay plots of any metric over training steps  
- **Figures**: Side-by-side display of all logged images
- **Config Diff**: Highlights configuration differences between runs

### Usage

#### Start the Dashboard
```bash
streamlit run spd/comparison/app.py
```

#### Compare Runs
1. Enter two W&B run paths in the sidebar:
   - Format: `entity/project/run_id`
   - Example Goodfire run: `goodfire/spd/uz2lmdjz`
   - Example SJCS run: `SJCS-SPD/spd/s326pufh`

2. Click "Fetch Run Data"

3. Explore the 4 tabs:
   - **📊 Metrics**: Numeric metrics comparison table
   - **📈 Training Curves**: Select and plot any metric
   - **🖼️ Figures**: View logged images side-by-side
   - **⚙️ Config**: See configuration differences

#### Cross-Team Comparison
The dashboard supports comparing runs from different W&B projects:
- **Goodfire** (external team): `goodfire/spd/[run_id]`
- **SJCS-SPD** (your team): `SJCS-SPD/spd/[run_id]`

#### Finding Runs
```bash
# List SJCS runs
python3 find_sjcs_projects.py

# List both teams' runs  
python3 list_both_teams_runs.py
```

## Implementation

Single file: `spd/comparison/app.py` (~520 lines)

Key functions:
- `fetch_run_data()`: Fetches all data from W&B API
- `compare_metrics()`: Creates comparison DataFrame with differences
- `plot_training_curves()`: Matplotlib visualization of training metrics
- `display_figures()`: Extracts image URLs for comparison
- `compare_configs()`: Highlights config differences

### Requirements

- W&B API key in `.env` file
- Dependencies: wandb, streamlit, matplotlib, pandas (via datasets)

---

## 2. Clustering Comparison Interface (`clustering_comparison.py`)

### Features

- **Side-by-side Clustering Comparison**: Compare different clustering methods on SPD runs
- **Dead Component Filtering**: Automatically filter components with low activation
- **MDL Clustering**: Minimum Description Length clustering implementation from spd/clustering
- **Performance Metrics**: Track wall time, memory usage, and theoretical FLOPs
- **ResidMLP Support**: Optimized for ResidMLP models from SPD runs

### Usage

#### Start the Interface
```bash
streamlit run spd/comparison/clustering_comparison.py --server.port 8510
```

#### Run Clustering Analysis

1. **Select Input Source**:
   - **W&B Path**: Use format like `wandb:goodfire/spd/runs/pziyck78`
   - **Local Path**: Use cached W&B runs like `wandb/84yirdkb/files`

2. **Configure Parameters**:
   - **Dead Component Threshold**: Components with max activation below this are filtered (default: 0.01)
   - **Batch Size**: Number of samples for generating activations (default: 1000)
   - **Device**: CPU or CUDA for computation
   - **MDL Alpha**: Complexity penalty for clustering (default: 1.0)
   - **MDL Iterations**: Number of clustering iterations (default: 140)

3. **Run Analysis**:
   - Click "Run Clustering" to start
   - View results in two-column layout:
     - **Left Column**: Placeholder for future custom clustering method
     - **Right Column**: MDL clustering results with metrics

### Component Files

```
spd/comparison/
├── clustering_comparison.py  # Main Streamlit interface
├── config.yaml              # Default configuration
├── spd_loader.py           # SPD run loading utilities
├── dead_filter.py          # Dead component filtering
├── mdl_clustering.py       # MDL clustering wrapper
└── metrics.py              # Performance metrics tracking
```

### Working Examples

**Important**: The interface requires **SPD runs** (with ComponentModel), not training checkpoints.

#### Local Cached SPD Runs
```python
# SPD runs with ComponentModel and final_config.yaml
"wandb/84yirdkb/files"   # TMS SPD run
"wandb/xawvyhq3/files"   # TMS SPD run  
"wandb/glbtwl6g/files"   # TMS SPD run

# Note: wandb/pziyck78/files, wandb/any9ekl9/files, wandb/6hk3uciu/files 
# are ResidMLP training checkpoints, NOT SPD runs - they won't work
```

#### W&B Paths
```python
# Direct W&B paths (requires authentication)
# These should be SPD runs, not training runs
# Check that the run has ComponentModel checkpoints
"wandb:project/spd/runs/run_id"  # Format for SPD runs
```

### Configuration

Edit `config.yaml` to change defaults:
```yaml
default_experiment: resid_mlp1
filter_dead_threshold: 0.01
mdl_config:
  activation_threshold: 0.01
  alpha: 1
  iters: 140
  merge_pair_sampling_method: "range"
  merge_pair_sampling_kwargs:
    threshold: 0.05
  pop_component_prob: 0
  filter_dead_threshold: 0.01
```

### Metrics Tracked

- **Wall Clock Time**: Total execution time
- **CPU Time**: Process CPU time
- **Memory Usage**: Peak memory consumption
- **Theoretical FLOPs**: Estimated floating-point operations
- **Clustering Quality**: Number of groups, MDL cost, merge iterations

### Implementation Notes

- Uses actual MDL clustering from `spd/clustering/merge.py`
- Filters dead components using `spd/clustering/activations.py` logic
- Loads models via `SPDRunInfo` and `ComponentModel` from spd.models
- No mocks or placeholders - all functionality is real
- Supports both local and W&B model loading
- Cross-platform compatible (no hardcoded paths)

### Requirements

- W&B API key in `.env` file (for W&B paths)
- Dependencies: torch, streamlit, wandb, psutil
- Clustering code from spd/clustering module