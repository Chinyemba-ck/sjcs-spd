#!/usr/bin/env python3
"""
Comprehensive Python 3.11 compatibility fix script.
Fixes ALL known issues in one pass.
"""

import os
import re
import sys
from pathlib import Path

def fix_file(filepath):
    """Fix all Python 3.11 compatibility issues in a file."""
    changes_made = False

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        # Fix 1: Override imports from typing
        if 'from typing import' in line and 'override' in line:
            # Split the import and remove override
            match = re.match(r'(\s*)from typing import (.*)', line)
            if match:
                indent = match.group(1)
                imports = match.group(2)
                # Remove override from the list
                import_list = [imp.strip() for imp in imports.split(',')]
                import_list = [imp for imp in import_list if imp != 'override']

                if import_list:
                    new_lines.append(f"{indent}from typing import {', '.join(import_list)}\n")
                new_lines.append(f"{indent}from typing_extensions import override\n")
                changes_made = True
                continue

        # Fix 2: Generic function syntax def func[T](...) -> def func(...)
        if re.search(r'def (\w+)\[.*?\]\(', line):
            new_line = re.sub(r'def (\w+)\[.*?\]\(', r'def \1(', line)
            new_lines.append(new_line)
            changes_made = True
            continue

        # Fix 3: Generic class syntax class Foo[T](Base) -> class Foo(Base)
        if re.search(r'class (\w+)\[.*?\]\(', line):
            new_line = re.sub(r'class (\w+)\[.*?\]\(', r'class \1(', line)
            new_lines.append(new_line)
            changes_made = True
            continue

        # Fix 4: RunInfo[T] -> RunInfo
        if 'RunInfo[' in line:
            new_line = re.sub(r'RunInfo\[[^\]]+\]', 'RunInfo', line)
            new_lines.append(new_line)
            changes_made = True
            continue

        # Fix 5: DataLoader[T] -> DataLoader
        if 'DataLoader[' in line and 'DataLoader[' in line:
            new_line = re.sub(r'DataLoader\[(\w+)\]', 'DataLoader', line)
            new_lines.append(new_line)
            changes_made = True
            continue

        # Fix 6: Corrupted @ override on separate lines
        if line.strip() == '@' and i + 1 < len(lines):
            next_line = lines[i + 1]
            if 'from typing_extensions import override' in next_line:
                # Skip both lines, they're corrupted
                skip_next = True
                changes_made = True
                continue

        # Fix 7: BaseModelType in function signatures
        if 'BaseModelType' in line and 'def ' in line:
            new_line = line.replace('BaseModelType', 'BaseModel')
            new_lines.append(new_line)
            changes_made = True
            continue

        # Fix 8: Return type BaseModelType -> BaseModel
        if '-> BaseModelType:' in line:
            new_line = line.replace('-> BaseModelType:', '-> BaseModel:')
            new_lines.append(new_line)
            changes_made = True
            continue

        # Default: keep line as-is
        new_lines.append(line)

    # Special handling for specific files
    if 'distributed_utils.py' in filepath:
        # Ensure ParamSpec and TypeVar are imported
        content = ''.join(new_lines)
        if 'Callable[P, T]' in content and 'ParamSpec' not in content:
            # Find the right place to add imports (after typing imports)
            for i, line in enumerate(new_lines):
                if 'from typing import' in line or 'import typing' in line:
                    # Add after this line
                    new_lines.insert(i + 1, 'from typing_extensions import ParamSpec\n')
                    new_lines.insert(i + 2, 'P = ParamSpec("P")\n')
                    new_lines.insert(i + 3, 'T = TypeVar("T")\n')
                    changes_made = True
                    break

    # Special fix for components.py - ensure @override decorator is on correct line
    if 'components.py' in filepath:
        # Fix the corrupted @override decorator
        fixed_lines = []
        i = 0
        while i < len(new_lines):
            line = new_lines[i]
            # Look for orphaned @override that should be a decorator
            if i > 0 and 'def forward' in line and '@override' not in new_lines[i-1]:
                # Check if there's already an @override nearby
                has_override = False
                for j in range(max(0, i-3), i):
                    if '@override' in new_lines[j]:
                        has_override = True
                        break
                if not has_override:
                    # Add @override decorator before the function
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(' ' * indent + '@override\n')
                    changes_made = True
            fixed_lines.append(line)
            i += 1
        new_lines = fixed_lines

    if changes_made:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    """Fix all Python files in the spd directory."""
    fixed_files = []

    # List of files we know need fixing (from our scan)
    critical_files = [
        'spd/eval.py',
        'spd/data.py',
        'spd/interfaces.py',
        'spd/models/component_model.py',
        'spd/models/components.py',
        'spd/models/sigmoids.py',
        'spd/utils/general_utils.py',
        'spd/utils/target_ci_solutions.py',
        'spd/utils/wandb_utils.py',
        'spd/utils/distributed_utils.py',
        'spd/utils/data_utils.py',
        'spd/experiments/tms/models.py',
        'spd/experiments/resid_mlp/models.py',
        'spd/experiments/resid_mlp/resid_mlp_dataset.py',
        'spd/experiments/ih/model.py'
    ]

    # Fix critical files first
    for filepath in critical_files:
        if os.path.exists(filepath):
            print(f"Fixing {filepath}...")
            if fix_file(filepath):
                fixed_files.append(filepath)

    # Also scan all other Python files for any missed issues
    for root, dirs, files in os.walk('spd'):
        # Skip clustering directory
        if 'clustering' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if filepath not in critical_files and filepath not in fixed_files:
                    if fix_file(filepath):
                        fixed_files.append(filepath)
                        print(f"Fixed additional file: {filepath}")

    print(f"\n=== SUMMARY ===")
    print(f"Fixed {len(fixed_files)} files:")
    for f in sorted(fixed_files):
        print(f"  - {f}")

    # Verify critical imports work
    print("\n=== VERIFICATION ===")
    try:
        # This will import the entire chain
        exec("from spd.experiments.lm import lm_decomposition")
        print("✓ Successfully imported lm_decomposition!")
        return True
    except ImportError as e:
        print(f"✗ Import still failing: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)