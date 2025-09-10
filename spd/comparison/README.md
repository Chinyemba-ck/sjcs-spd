# SPD Run Comparison Dashboard

A Streamlit dashboard for comparing SPD (Sparse Parameter Decomposition) runs between teams via W&B.

## Features

- **Metrics Comparison**: Side-by-side final metrics with percentage differences
- **Training Curves**: Overlay plots of any metric over training steps  
- **Figures**: Side-by-side display of all logged images
- **Config Diff**: Highlights configuration differences between runs

## Usage

### Start the Dashboard
```bash
streamlit run spd/comparison/app.py
```

### Compare Runs
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

### Cross-Team Comparison
The dashboard supports comparing runs from different W&B projects:
- **Goodfire** (external team): `goodfire/spd/[run_id]`
- **SJCS-SPD** (your team): `SJCS-SPD/spd/[run_id]`

### Finding Runs
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

## Requirements

- W&B API key in `.env` file
- Dependencies: wandb, streamlit, matplotlib, pandas (via datasets)