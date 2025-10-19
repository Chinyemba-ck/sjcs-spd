#!/bin/bash
# Deployment script for SmolLM2 3-layer attention decomposition on 2×H200

set -e  # Exit on any error

POD_SSH="ssh -i ~/.ssh/runpod_key -p 17296 root@198.145.108.46"

echo "=== Deploying SmolLM2 Attention Decomposition to rising_bronze_dolphin ==="
echo "Pod: vxwjz8a6waw631"
echo "GPUs: 2× NVIDIA H200"
echo ""

# Load credentials from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "ERROR: .env file not found!"
    exit 1
fi

# Verify required credentials
if [ -z "$WANDB_API_KEY" ] || [ -z "$HF_TOKEN" ]; then
    echo "ERROR: WANDB_API_KEY or HF_TOKEN not set in .env"
    exit 1
fi

echo "✓ Credentials loaded from .env"
echo ""

# Clone repository
echo "Step 1: Cloning repository..."
$POD_SSH "cd /workspace && \
    git clone https://github.com/Chinyemba-ck/sjcs-spd.git && \
    cd sjcs-spd && \
    git checkout sean && \
    echo '✓ Repository cloned on branch sean'"

echo ""
echo "Step 2: Installing dependencies..."
$POD_SSH "cd /workspace/sjcs-spd && \
    pip install --break-system-packages -e . && \
    echo '✓ Dependencies installed'"

echo ""
echo "Step 3: Configuring credentials..."
$POD_SSH "cd /workspace/sjcs-spd && \
    wandb login $WANDB_API_KEY && \
    mkdir -p ~/.huggingface && \
    echo '$HF_TOKEN' > ~/.huggingface/token && \
    echo '✓ WandB and HuggingFace credentials configured'"

echo ""
echo "Step 4: Verifying PyTorch GPU detection..."
$POD_SSH "python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\"); [print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\") for i in range(torch.cuda.device_count())]'"

echo ""
echo "Step 5: Launching training with torchrun..."
$POD_SSH "cd /workspace/sjcs-spd && \
    nohup torchrun --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=29500 \
        spd/experiments/lm/lm_decomposition.py \
        spd/experiments/lm/smollm2_135m_3layer_attn_config.yaml \
    > training_attn_2xh200.log 2>&1 &"

echo ""
echo "✓ Training launched in background!"
echo ""
echo "Monitor training:"
echo "  ssh -i ~/.ssh/runpod_key -p 17296 root@198.145.108.46 'tail -f /workspace/sjcs-spd/training_attn_2xh200.log'"
echo ""
echo "Check GPU usage:"
echo "  ssh -i ~/.ssh/runpod_key -p 17296 root@198.145.108.46 'nvidia-smi'"
echo ""
echo "WandB: https://wandb.ai/SJCS-SPD/smollm2-spd"
