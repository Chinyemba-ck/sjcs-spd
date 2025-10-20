#!/bin/bash
set -e  # Exit on error

echo "========================================="
echo "SPD 2×H200 Deployment Script"
echo "========================================="

# Pod connection details
POD_IP="50.145.48.118"
POD_PORT="16179"
POD_ID="875qji18gxizxs"

# Load credentials from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

echo ""
echo "Step 1: Creating deployment archive..."
echo "========================================="

# Create a clean tarball excluding .git, wandb, checkpoints, etc.
tar -czf sjcs-spd-deploy.tar.gz \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='wandb' \
    --exclude='.checkpoints' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.data' \
    --exclude='.env' \
    --exclude='tmp_checkpoint_verification' \
    --exclude='.git-rewrite' \
    --exclude='*.tar.gz' \
    .

echo "✓ Archive created: sjcs-spd-deploy.tar.gz"

echo ""
echo "Step 2: Uploading code to pod..."
echo "========================================="

# Upload the tarball (disable strict host key checking for new pod)
scp -o StrictHostKeyChecking=no -P $POD_PORT sjcs-spd-deploy.tar.gz root@$POD_IP:/workspace/

echo "✓ Code uploaded successfully"

echo ""
echo "Step 3: Setting up environment on pod..."
echo "========================================="

# SSH into pod and set up environment
ssh -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_IP bash << 'REMOTE_SETUP'
set -e

cd /workspace

# Extract code
echo "Extracting code..."
mkdir -p sjcs-spd
cd sjcs-spd
tar -xzf ../sjcs-spd-deploy.tar.gz

# Set up Python environment
echo "Setting up Python environment..."
pip install --upgrade pip
pip install -e .

# Configure WandB
echo "Configuring WandB..."
wandb login WANDB_API_KEY_PLACEHOLDER

# Configure HuggingFace
echo "Configuring HuggingFace..."
mkdir -p ~/.huggingface
echo "HF_TOKEN_PLACEHOLDER" > ~/.huggingface/token

# Verify GPU setup
echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Check PyTorch can see both GPUs
python3 << EOF
import torch
print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo ""
echo "✓ Environment setup complete!"

REMOTE_SETUP

# Replace placeholders with actual credentials
ssh -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_IP "cd /workspace/sjcs-spd && wandb login $WANDB_API_KEY && echo '$HF_TOKEN' > ~/.huggingface/token"

echo ""
echo "Step 4: Launching distributed training..."
echo "========================================="

# Launch training with torchrun
ssh -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_IP bash << REMOTE_TRAIN
set -e

cd /workspace/sjcs-spd

# Set WandB environment variables
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_ENTITY="$WANDB_ENTITY"

# Set HuggingFace token
export HF_TOKEN="$HF_TOKEN"

# Launch DDP training with torchrun
echo "Starting training with torchrun (2×H200)..."
echo "Config: spd/experiments/lm/smollm2_135m_3layer_config.yaml"
echo "Batch size: 64 per GPU (effective batch = 128)"
echo ""

torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    spd/experiments/lm/lm_decomposition.py \
    spd/experiments/lm/smollm2_135m_3layer_config.yaml \
    2>&1 | tee training_output.log

REMOTE_TRAIN

echo ""
echo "========================================="
echo "✓ Deployment and training launch complete!"
echo "========================================="
echo ""
echo "Useful commands:"
echo "  - Monitor training: ssh -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_IP 'tail -f /workspace/sjcs-spd/training_output.log'"
echo "  - Check GPU usage: ssh -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_IP 'watch -n 1 nvidia-smi'"
echo "  - SSH into pod: ssh -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_IP"
echo ""
