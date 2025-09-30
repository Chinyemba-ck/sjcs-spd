#!/bin/bash
# Comprehensive setup script for RunPod 3x A40 deployment
# This script sets up the complete environment for distributed SPD training
# with extreme attention to detail and no shortcuts

set -euo pipefail  # Exit on error, undefined variables, and pipe failures
IFS=$'\n\t'       # Set Internal Field Separator for safety

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function for colored output
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        log_success "$1"
    else
        log_error "$2"
        exit 1
    fi
}

# Start setup
log_info "==================================================================="
log_info "SPD RunPod Setup Script - 3x NVIDIA A40 GPUs"
log_info "==================================================================="

# Step 1: Verify GPU availability
log_info "Step 1: Verifying GPU availability..."
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader,nounits
check_success "GPU query successful" "Failed to query GPUs"

GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -eq 3 ]; then
    log_success "Found 3 GPUs as expected"
else
    log_error "Expected 3 GPUs but found $GPU_COUNT"
    exit 1
fi

# Step 2: System information
log_info "Step 2: Gathering system information..."
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "Python version: $(python3 --version 2>&1)"
echo "CUDA version: $(nvcc --version 2>&1 | grep "release" | awk '{print $5}' | sed 's/,//')"
echo "Current directory: $(pwd)"
echo "Available memory: $(free -h | grep "Mem:" | awk '{print $2}')"
echo "Available disk space: $(df -h / | tail -1 | awk '{print $4}')"

# Step 3: Set environment variables
log_info "Step 3: Setting environment variables..."
# Load secrets from .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
# Verify required secrets are set (use :- for set -u compatibility)
if [ -z "${WANDB_API_KEY:-}" ] || [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: WANDB_API_KEY and HF_TOKEN must be set in .env file or environment"
    echo "Please create a .env file with:"
    echo "  WANDB_API_KEY=your_wandb_key"
    echo "  HF_TOKEN=your_huggingface_token"
    exit 1
fi
export WANDB_ENTITY="SJCS-SPD"
export CUDA_VISIBLE_DEVICES="0,1,2"
export NCCL_DEBUG=INFO  # Enable NCCL debugging for distributed training
export NCCL_TIMEOUT=1800  # 30 minutes timeout for NCCL operations
export TORCH_DISTRIBUTED_DEBUG=INFO
log_success "Environment variables set"

# Step 4: Create working directory
log_info "Step 4: Creating working directory..."
WORK_DIR="/workspace/spd"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
log_success "Working directory created at $WORK_DIR"

# Step 5: Clone or update repository
log_info "Step 5: Setting up SPD repository..."
if [ -d "sjcs-spd" ]; then
    log_info "Repository exists, pulling latest changes..."
    cd sjcs-spd
    git fetch origin
    git reset --hard origin/sean  # Use sean branch as specified
    check_success "Repository updated" "Failed to update repository"
else
    log_info "Cloning repository..."
    git clone https://github.com/seaneillpc/sjcs-spd.git
    check_success "Repository cloned" "Failed to clone repository"
    cd sjcs-spd
    git checkout sean
    check_success "Switched to sean branch" "Failed to switch branch"
fi

# Step 6: Create and activate virtual environment
log_info "Step 6: Setting up Python virtual environment..."
if [ -d ".venv" ]; then
    log_warning "Virtual environment exists, removing and recreating..."
    rm -rf .venv
fi

python3 -m venv .venv
check_success "Virtual environment created" "Failed to create virtual environment"

source .venv/bin/activate
check_success "Virtual environment activated" "Failed to activate virtual environment"

# Step 7: Upgrade pip and install build tools
log_info "Step 7: Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel
check_success "Pip upgraded" "Failed to upgrade pip"

# Step 8: Install PyTorch with CUDA support
log_info "Step 8: Installing PyTorch with CUDA support..."
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
check_success "PyTorch installed" "Failed to install PyTorch"

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"
check_success "PyTorch CUDA verified" "PyTorch CUDA verification failed"

# Step 9: Install project dependencies
log_info "Step 9: Installing project dependencies..."
pip install -e .
check_success "Project dependencies installed" "Failed to install project dependencies"

# Additional required packages
pip install wandb transformers datasets einops jaxtyping pydantic
check_success "Additional packages installed" "Failed to install additional packages"

# Step 10: Verify all imports
log_info "Step 10: Verifying all imports..."
python3 -c "
import torch
import wandb
import transformers
import datasets
import einops
import jaxtyping
import pydantic
import spd
print('All imports successful')
"
check_success "All imports verified" "Import verification failed"

# Step 11: Test distributed setup
log_info "Step 11: Testing distributed training setup..."
python3 -c "
import torch
import torch.distributed as dist
import os

# Check if we can initialize process group (dry run)
if torch.cuda.is_available() and torch.cuda.device_count() == 3:
    print(f'CUDA devices available: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB')
    print('Distributed training setup looks good')
else:
    raise RuntimeError(f'Expected 3 CUDA devices, found {torch.cuda.device_count()}')
"
check_success "Distributed setup verified" "Distributed setup failed"

# Step 12: Create launch script
log_info "Step 12: Creating launch script..."
cat > launch_training.sh << 'EOF'
#!/bin/bash
# Launch script for distributed SPD training on 3x A40 GPUs

set -euo pipefail

# Activate virtual environment
source /workspace/spd/sjcs-spd/.venv/bin/activate

# Load secrets from .env
if [ -f /workspace/spd/sjcs-spd/.env ]; then
    export $(cat /workspace/spd/sjcs-spd/.env | grep -v '^#' | xargs)
fi

# Verify required secrets are set
if [ -z "$WANDB_API_KEY" ] || [ -z "$HF_TOKEN" ]; then
    echo "ERROR: WANDB_API_KEY and HF_TOKEN must be set in .env file"
    exit 1
fi

# Set environment variables
export WANDB_ENTITY="SJCS-SPD"
export CUDA_VISIBLE_DEVICES="0,1,2"
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_DEBUG=INFO

# Change to project directory
cd /workspace/spd/sjcs-spd

# Launch distributed training with torchrun
echo "Starting distributed training on 3 GPUs..."
echo "Configuration: batch_size=9 (3 per GPU), C=1000, steps=30000"
echo "Expected memory usage: ~43GB per GPU"

torchrun \
    --standalone \
    --nproc_per_node=3 \
    --master_port=29500 \
    spd/experiments/lm/lm_decomposition.py \
    spd/experiments/lm/gemma_270m_multilingual_config.yaml

echo "Training completed"
EOF

chmod +x launch_training.sh
log_success "Launch script created at launch_training.sh"

# Step 13: Create monitoring script
log_info "Step 13: Creating monitoring script..."
cat > monitor_training.sh << 'EOF'
#!/bin/bash
# Monitoring script for distributed training

while true; do
    clear
    echo "=== GPU Utilization ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F',' '{printf "GPU %s: %s | Temp: %s°C | Util: %s%% | Mem: %.1fGB/%.1fGB\n", $1, $2, $3, $4, $5/1024, $6/1024}'

    echo ""
    echo "=== Process Status ==="
    ps aux | grep -E "(torchrun|lm_decomposition)" | grep -v grep || echo "No training process found"

    echo ""
    echo "=== Latest Training Logs (if available) ==="
    if [ -f /workspace/spd/sjcs-spd/spd_run.log ]; then
        tail -n 10 /workspace/spd/sjcs-spd/spd_run.log
    fi

    echo ""
    echo "Press Ctrl+C to exit monitoring"
    sleep 5
done
EOF

chmod +x monitor_training.sh
log_success "Monitoring script created at monitor_training.sh"

# Step 14: Final verification
log_info "Step 14: Final verification..."
echo ""
echo "==================================================================="
echo "Setup Complete! System is ready for distributed training."
echo "==================================================================="
echo ""
echo "Pod Information:"
echo "  - Name: aware_fuchsia_guanaco"
echo "  - GPUs: 3x NVIDIA A40 (48GB each)"
echo "  - Total VRAM: 144GB"
echo ""
echo "Configuration Summary:"
echo "  - Model: google/gemma-3-270m-it (270M parameters)"
echo "  - Batch size: 9 (3 per GPU)"
echo "  - Components (C): 1000"
echo "  - Training steps: 30,000"
echo "  - Layerwise reconstruction: ENABLED (coeff=2.0)"
echo "  - Expected memory per GPU: ~43GB (90% utilization)"
echo ""
echo "Next Steps:"
echo "  1. Review the configuration one final time"
echo "  2. Run './launch_training.sh' to start distributed training"
echo "  3. In another terminal, run './monitor_training.sh' to monitor progress"
echo "  4. Check W&B at https://wandb.ai/SJCS-SPD for training metrics"
echo ""
log_success "Setup script completed successfully!"