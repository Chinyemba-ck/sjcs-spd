#!/usr/bin/env python3
"""List available SPD runs from W&B."""

import wandb
from dotenv import load_dotenv

load_dotenv()

api = wandb.Api()
runs = api.runs("goodfire/spd", per_page=20)

print("Available SPD runs in goodfire/spd:\n")
for i, run in enumerate(runs, 1):
    print(f"{i}. goodfire/spd/{run.id}")
    print(f"   Name: {run.name}")
    print(f"   State: {run.state}")
    print(f"   Created: {run.created_at}")
    print()