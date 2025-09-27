#!/usr/bin/env python3
"""List SPD runs from both Goodfire and SJCS teams."""

import wandb
from dotenv import load_dotenv

load_dotenv()

api = wandb.Api()

print("=" * 60)
print("COMPARING RUNS FROM TWO TEAMS")
print("=" * 60)

# List Goodfire runs
print("\n🔥 GOODFIRE TEAM (External/Original):")
print("-" * 40)
try:
    runs = api.runs("goodfire/spd", per_page=5)
    for i, run in enumerate(runs, 1):
        print(f"{i}. goodfire/spd/{run.id}")
        print(f"   Name: {run.name}")
        print(f"   State: {run.state}")
        print()
except Exception as e:
    print(f"Error accessing goodfire/spd: {e}")

# List SJCS team runs - try different possible project names
print("\n🚀 SJCS TEAM (Your Fork):")
print("-" * 40)

possible_projects = ["SJCS-SPD/spd", "SJCS/spd", "SJCS-SPD/sjcs-spd"]

for project in possible_projects:
    try:
        runs = api.runs(project, per_page=5)
        run_list = list(runs)
        if run_list:
            print(f"Found runs in {project}:")
            for i, run in enumerate(run_list, 1):
                print(f"{i}. {project}/{run.id}")
                print(f"   Name: {run.name}")
                print(f"   State: {run.state}")
                print()
            break
    except Exception:
        continue
else:
    print("Could not find SJCS-SPD runs. Trying to list all SJCS-SPD projects...")
    try:
        # Try to get user info to find their projects
        user = api.viewer()
        print(f"Logged in as: {user.username}")
        print(f"Entity: {user.entity}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 60)
print("HOW TO USE THE COMPARISON DASHBOARD:")
print("=" * 60)
print("1. Run: streamlit run spd/comparison/app.py")
print("2. Enter Run 1: goodfire/spd/[run_id]  (external team)")
print("3. Enter Run 2: SJCS-SPD/[project]/[run_id]  (your team)")
print("4. Click 'Fetch Run Data' to compare!")