#!/usr/bin/env python3
"""Find SJCS-SPD project names."""

import wandb
from dotenv import load_dotenv

load_dotenv()

api = wandb.Api()

print("Looking for SJCS-SPD projects...")
print("=" * 60)

# Try different entity/project combinations
test_combinations = [
    "SJCS/spd",
    "SJCS-SPD/spd", 
    "SJCS/sjcs-spd",
    "SJCS-SPD/sjcs-spd",
    "seanesla/spd",  # Your username
    "seanesla/sjcs-spd"
]

for project_path in test_combinations:
    try:
        runs = api.runs(project_path, per_page=1)
        run_list = list(runs)
        if run_list:
            print(f"\n✓ Found project: {project_path}")
            print(f"  Example run: {project_path}/{run_list[0].id}")
            print(f"  Run name: {run_list[0].name}")
    except Exception:
        pass

print("\n" + "=" * 60)
print("To compare runs from both teams:")
print("1. Run: streamlit run spd/comparison/app.py") 
print("2. Enter a Goodfire run: goodfire/spd/[run_id]")
print("3. Enter your team's run: [found_project_path]/[run_id]")