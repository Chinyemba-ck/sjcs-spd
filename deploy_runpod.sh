#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "RunPod Deployment Script"
echo "SmolLM2 Attention C=2000 Fixed"
echo "=========================================="

# RunPod connection details
RUNPOD_IP="213.181.122.225"
RUNPOD_PORT="17781"
POD_ID="vumnrfnp6q63zl"

echo ""
echo "Connecting to RunPod: $RUNPOD_IP:$RUNPOD_PORT"
echo "Pod ID: $POD_ID"
echo ""

# Create setup script that will run on RunPod
cat > /tmp/runpod_setup.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "Setting up RunPod environment..."
echo "=========================================="

cd /workspace

# Check GPUs
echo ""
echo "Checking GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Clone repo (or pull if exists)
if [ -d "sjcs-spd" ]; then
    echo "Repository exists, pulling latest..."
    cd sjcs-spd
    git fetch origin
    git checkout sean
    git pull origin sean
else
    echo "Cloning repository..."
    git clone https://github.com/Chinyemba-ck/sjcs-spd.git
    cd sjcs-spd
    git checkout sean
fi

# Verify our config exists
echo ""
echo "Verifying config file..."
if [ ! -f "spd/experiments/lm/smollm2_135m_2layer_attn_fixed.yaml" ]; then
    echo "ERROR: Config file not found!"
    exit 1
fi
echo "✓ Config file found"

# Show config details
echo ""
echo "Config details:"
grep "^C:" spd/experiments/lm/smollm2_135m_2layer_attn_fixed.yaml
grep "^importance_minimality_coeff:" spd/experiments/lm/smollm2_135m_2layer_attn_fixed.yaml
echo ""

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment and install dependencies
echo "Creating virtual environment and installing dependencies..."
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Set up WandB credentials (need to be provided)
echo ""
echo "=========================================="
echo "WandB Setup Required"
echo "=========================================="
echo "Please provide your WandB API key:"
read -s WANDB_API_KEY
export WANDB_API_KEY=$WANDB_API_KEY
wandb login $WANDB_API_KEY

echo ""
echo "=========================================="
echo "Environment Setup Complete"
echo "=========================================="
echo ""
echo "Ready to launch training!"
echo ""

SCRIPT_EOF

# Copy setup script to RunPod
echo "Copying setup script to RunPod..."
scp -P $RUNPOD_PORT /tmp/runpod_setup.sh root@$RUNPOD_IP:/tmp/

# SSH into RunPod and run setup
echo ""
echo "Running setup on RunPod..."
ssh -p $RUNPOD_PORT root@$RUNPOD_IP 'bash /tmp/runpod_setup.sh'

# Create training launch script
cat > /tmp/launch_training.sh << 'TRAIN_EOF'
#!/bin/bash
set -e

cd /workspace/sjcs-spd
source .venv/bin/activate

echo "=========================================="
echo "Launching Training with torchrun"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Config: smollm2_135m_2layer_attn_fixed.yaml"
echo "  - C: 2000 (24,000 total components)"
echo "  - importance_minimality_coeff: 0.0008"
echo "  - GPUs: 2× H100"
echo "  - Batch size: 64 (32 per GPU)"
echo "  - Steps: 30,000"
echo ""

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1

# Launch with torchrun for DDP
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    spd/experiments/lm/lm_decomposition.py \
    spd/experiments/lm/smollm2_135m_2layer_attn_fixed.yaml

echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="

TRAIN_EOF

# Copy training script to RunPod
echo "Copying training launch script to RunPod..."
scp -P $RUNPOD_PORT /tmp/launch_training.sh root@$RUNPOD_IP:/tmp/

echo ""
echo "=========================================="
echo "Deployment Complete"
echo "=========================================="
echo ""
echo "To launch training, SSH into RunPod and run:"
echo "  ssh -p $RUNPOD_PORT root@$RUNPOD_IP"
echo "  bash /tmp/launch_training.sh"
echo ""
echo "To monitor training:"
echo "  - WandB: https://wandb.ai/SJCS-SPD/smollm2-spd"
echo "  - Watch for gradient norms (should stay < 0.05)"
echo "  - Watch for KL gap (should stay negative)"
echo "  - Watch for component survival (should be > 5%)"
echo ""
