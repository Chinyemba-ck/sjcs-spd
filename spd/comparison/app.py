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