# SPD: Sparse Parameter Decomposition

A research framework for decomposing neural network parameters into sparse, interpretable components using stochastic optimization.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

SPD decomposes pretrained neural networks into collections of sparse components that can be analyzed independently. This enables fine-grained interpretability analysis by identifying which components contribute to specific behaviors or outputs.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/goodfire-ai/spd.git
cd spd

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.2.0
- CUDA-capable GPU (recommended for language model experiments)

### Setup

Create a `.env` file in the repository root:

```bash
WANDB_API_KEY=your_wandb_key_here
WANDB_ENTITY=your_wandb_username
```

Get your WandB API key from https://wandb.ai/authorize

### Run Your First Experiment

```bash
# Run a toy model experiment (completes in ~4 minutes)
spd-run --experiments tms_5-2

# View results at the WandB URL printed to console
```

This runs a Toy Model of Superposition (TMS) experiment with 5 features and 2 hidden dimensions. Results include:
- Component sparsity patterns
- Reconstruction quality metrics
- Importance scores for each component

## Available Experiments

SPD supports three types of experiments:

### Toy Model of Superposition (TMS)
Fast experiments on synthetic models designed to exhibit superposition:
- `tms_5-2` - 5 features, 2 hidden dims (~4 min)
- `tms_5-2-id` - 5 features, 2 hidden dims with identity constraint (~4 min)
- `tms_40-10` - 40 features, 10 hidden dims (~5 min)
- `tms_40-10-id` - 40 features, 10 hidden dims with identity constraint (~5 min)

### Residual MLP
Decomposition of residual multilayer perceptrons:
- `resid_mlp1` - 1 layer (~3 min)
- `resid_mlp2` - 2 layers (~5 min)
- `resid_mlp3` - 3 layers (~60 min)

### Language Models
Decomposition of pretrained transformer models:
- `ss_gpt2` - GPT-2 on Simple Stories dataset (~60 min)
- `ss_gpt2_simple` - Simplified GPT-2 variant (~5.5 hours)
- `ss_gpt2_simple_noln` - GPT-2 without layer normalization (~5.5 hours)
- `gpt2` - Standard GPT-2 model (~60 min)
- `ss_llama` - Llama on Simple Stories (multi-GPU, ~37 hours)
- `ss_llama_single` - Llama single GPU (~94 hours)
- `smollm2_135m_3layer` - SmolLM2-135M 3-layer decomposition (~12.5 hours)
- `smollm2_135m_5layer` - SmolLM2-135M 5-layer decomposition (~20.8 hours)

List all experiments:
```bash
spd-run --help
```

## Common Commands

### Running Experiments

```bash
# Run single experiment
spd-run --experiments tms_5-2

# Run multiple experiments
spd-run --experiments tms_5-2,resid_mlp1,ss_gpt2

# Run all experiments (not recommended - will take days)
spd-run
```

### Hyperparameter Sweeps

```bash
# Run a sweep with 4 parallel agents
spd-run --experiments tms_5-2 --sweep --n-agents 4

# Use custom sweep parameters
spd-run --experiments tms_5-2 --sweep custom_sweep.yaml --n-agents 2

# Run sweep on CPU instead of GPU
spd-run --experiments tms_5-2 --sweep --n-agents 2 --cpu
```

Sweep parameters are defined in `spd/scripts/sweep_params.yaml`. See `spd/scripts/sweep_params.yaml.example` for the parameter structure.

### Clustering Analysis

```bash
# Run clustering analysis on SPD results
spd-cluster --run_path wandb:project/runs/run_id --method kmeans --n_clusters 10

# Interactive comparison of clustering methods
streamlit run spd/comparison/clustering_comparison.py --server.port 8510
```

### Model Comparison

Compare geometric similarities between two SPD models:

```bash
python spd/scripts/compare_models/compare_models.py \
  --current_model_path="wandb:project/runs/run1" \
  --reference_model_path="wandb:project/runs/run2"
```

See `spd/scripts/compare_models/README.md` for details.

## Project Structure

```
spd/
├── experiments/          # Experiment implementations (TMS, ResidualMLP, LM)
│   ├── tms/             # Toy Model of Superposition
│   ├── resid_mlp/       # Residual MLP experiments
│   └── lm/              # Language model experiments
├── models/              # Core model classes
│   ├── component_model.py  # Main ComponentModel wrapper
│   └── components.py       # Component type definitions
├── clustering/          # Component clustering analysis
│   ├── scripts/         # CLI tools
│   └── experiments/     # Clustering experiments
├── comparison/          # Interactive comparison tools
│   └── clustering_comparison.py  # Streamlit app
├── scripts/             # Utility scripts
│   ├── run.py           # Main spd-run CLI
│   ├── compare_models/  # Model comparison tools
│   └── deployment/      # RunPod/SLURM deployment
├── utils/               # Shared utilities
│   ├── distributed_utils.py  # Distributed training helpers
│   ├── losses.py             # SPD loss functions
│   └── metrics.py            # Evaluation metrics
├── configs.py           # Pydantic configuration classes
├── registry.py          # Experiment registry
├── run_spd.py          # Core SPD optimization loop
├── losses.py           # Loss function implementations
├── metrics.py          # Metric computation
└── figures.py          # Visualization utilities
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
make install-dev

# This installs:
# - pytest (testing)
# - ruff (linting & formatting)
# - basedpyright (type checking)
# - pre-commit hooks
```

### Code Quality

```bash
# Run all checks (type checking, linting, formatting)
make check

# Run only type checking
make type

# Run only linting and formatting
make format
```

### Testing

```bash
# Run tests (excludes slow tests)
make test

# Run all tests including slow ones
make test-all

# Run tests with coverage report
make coverage  # Outputs to docs/coverage/

# Run specific test file
python -m pytest tests/test_specific.py

# Run specific test function
python -m pytest tests/test_specific.py::test_function
```

### Contributing

Before submitting a pull request:

1. Run `make check` and ensure all checks pass
2. Review your diff carefully
3. Merge latest changes from the `dev` branch
4. Follow the PR template in `.github/pull_request_template.md`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Code Style

- Use type annotations (jaxtyping for tensors)
- Prefer `einops` for tensor operations
- Assert assumptions liberally (fail fast)
- Prioritize readable code over clever code

See [STYLE.md](STYLE.md) for complete style guide.

## Research Papers

This repository implements methods from two research papers:

### Stochastic Parameter Decomposition (SPD)

The core SPD framework with stochastic masking and optimization techniques.

📄 [Read the paper](papers/Stochastic_Parameter_Decomposition/spd_paper.md)

**Key contributions:**
- Differentiable sparse decomposition via stochastic masking
- Multi-objective optimization balancing faithfulness and sparsity
- Scalable to large language models

### Attribution-based Parameter Decomposition (APD)

Conceptual foundations and theoretical framework for parameter decomposition.

📄 [Read the paper](papers/Attribution_based_Parameter_Decomposition/apd_paper.md)

**Key contributions:**
- Linear parameter decomposition framework
- Attribution-based component identification
- Theoretical analysis of decomposition properties

### Citation

If you use this code in your research, please cite:

```bibtex
@software{spd2024,
  title={Sparse Parameter Decomposition},
  author={Goodfire AI},
  year={2024},
  url={https://github.com/goodfire-ai/spd}
}
```

## Architecture Overview

### Core Concepts

**ComponentModel**: Wraps a pretrained "target" model and learns to decompose its parameters into sparse components. Each component represents a subset of the original parameters that can be analyzed independently.

**Stochastic Masking**: Components use probabilistic masks during training that become deterministic at inference. This enables differentiable sparsity optimization.

**Multi-objective Loss**: Balances three objectives:
- **Faithfulness**: Component model outputs match target model outputs
- **Reconstruction**: Components can reconstruct original parameters
- **Sparsity**: Minimize number of active components

### Training Flow

1. Load pretrained target model (e.g., GPT-2)
2. Wrap in ComponentModel with specified target modules
3. Optimize component masks using combined loss
4. Analyze resulting sparse decomposition

See `spd/run_spd.py` for the core optimization loop.

## Deployment

### RunPod GPU Deployment

SPD supports deployment on RunPod cloud GPUs:

1. Configure RunPod credentials in `.env`
2. Use deployment scripts in `spd/scripts/deployment/`
3. Monitor training via WandB

### SLURM Cluster

For hyperparameter sweeps on SLURM clusters:

```bash
# Submit sweep job array
spd-run --experiments tms_5-2 --sweep --n-agents 8
```

This creates a WandB sweep and launches SLURM agents to run it.

### Distributed Training

Language model experiments support PyTorch Distributed Data Parallel (DDP):

- Automatically detects world size and rank
- Handles gradient synchronization
- Supports both single-GPU and multi-GPU training

See `spd/utils/distributed_utils.py` for implementation.

## Known Issues

- **NCCL Deadlock**: Fixed in v0.0.1 - world_size=1 no longer triggers collective operation deadlocks
- **Python Version**: Requires Python 3.10+. Python 3.13 is supported.
- **PyTorch Version**: Minimum torch 2.2.0 (relaxed from 2.6.0 for broader compatibility)

## Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines and PR process
- **[STYLE.md](STYLE.md)**: Code style guide and best practices
- **[papers/](papers/)**: Research paper implementations and figures
- **[spd/comparison/README.md](spd/comparison/README.md)**: Clustering comparison tool
- **[spd/scripts/compare_models/README.md](spd/scripts/compare_models/README.md)**: Model comparison tool

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/goodfire-ai/spd/issues)
- **Discussions**: [GitHub Discussions](https://github.com/goodfire-ai/spd/discussions)

Maintained by [Goodfire AI](https://goodfire.ai)
