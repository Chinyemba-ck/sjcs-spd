#!/usr/bin/env python3
"""Fix Python 3.11 compatibility issues in SPD codebase"""

import os
import re

def fix_file(filepath, replacements):
    """Apply replacements to a file"""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} does not exist")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
    else:
        print(f"No changes needed: {filepath}")

# Fix components.py - override import
fix_file('spd/models/components.py', [
    (r'from typing import(.*)override', r'from typing import\1\nfrom typing_extensions import override')
])

# Fix identity_insertion.py and component_model.py - Conv1D import
for file in ['spd/identity_insertion.py', 'spd/models/component_model.py']:
    fix_file(file, [
        (r'from transformers\.modeling_utils import Conv1D',
         r'from transformers.pytorch_utils import Conv1D')
    ])

# Fix interfaces.py - Remove generic syntax
fix_file('spd/interfaces.py', [
    (r'class RunInfo\[T\]\(ABC\):', r'class RunInfo(ABC):'),
    (r'config: T', r'config: Any'),
    # Add Any import if not present
    (r'from typing import (.*)', lambda m: f"from typing import {m.group(1)}, Any" if 'Any' not in m.group(1) else m.group(0))
])

# Fix general_utils.py - Remove all generic syntax
fix_file('spd/utils/general_utils.py', [
    # Fix load_config function
    (r'def load_config\[T: BaseModel\]\(\s*config_path: Path \| str,\s*config_cls: type\[T\]\s*\) -> T:',
     r'def load_config(config_path: Path | str, config_cls: type[BaseModel]) -> BaseModel:'),
    # Fix save_config function
    (r'def save_config\[T: BaseModel\]\(config: T, config_path: Path \| str\) -> None:',
     r'def save_config(config: BaseModel, config_path: Path | str) -> None:'),
    # Fix other generic functions if present
    (r'def (\w+)\[T\]\((.*?)\) -> (.*?):', r'def \1(\2) -> \3:'),
    (r'def (\w+)\[T: (\w+)\]\((.*?)\) -> (.*?):', r'def \1(\3) -> \4:'),
])

# Fix data.py - Remove generic from loop_dataloader
fix_file('spd/data.py', [
    (r'def loop_dataloader\[T\]\(dl: DataLoader\[T\]\):',
     r'def loop_dataloader(dl: DataLoader):'),
    (r'-> Iterator\[T\]:', r'-> Iterator:')
])

# Fix distributed_utils.py - Remove all generic syntax
fix_file('spd/utils/distributed_utils.py', [
    # Remove [**P, T] syntax
    (r'def (\w+)\[\*\*P, T\]\((.*?)\) -> (.*?):', r'def \1(\2) -> \3:'),
    (r'def (\w+)\[T\]\((.*?)\) -> (.*?):', r'def \1(\2) -> \3:'),
    # Fix Callable with ParamSpec
    (r'Callable\[\[.*?\], T\]', r'Callable'),
    (r'func: Callable\[\*\*P, T\]', r'func: Callable'),
    # Remove ParamSpec import if present
    (r'from typing import.*ParamSpec.*\n', r''),
])

# Also need to install typing_extensions if not present
print("\nInstalling typing_extensions...")
os.system("pip install typing_extensions")

print("\nPython 3.11 compatibility fixes applied!")