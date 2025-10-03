# SmolLM2-135M 5-Layer SPD Deployment Checklist

**Training Configuration:**
- Model: SmolLM2-135M-Instruct
- Layers: 13-17 (5 layers, 15 modules)
- Steps: 30,000
- Expected duration: 19-21 hours
- Expected cost: $134-145

**Target Pod:**
- 4× A100-SXM4-80GB
- Secure Cloud (on-demand)
- Price: $6.96/hr
- Storage: 100GB

---

## PRE-DEPLOYMENT (Local Machine)

### ☐ 1. Verify Local Files
```bash
cd /Users/seane/Documents/Github/sjcs-spd

# Check config exists
ls -la spd/experiments/lm/smollm2_135m_5layer_config.yaml

# Check registry entry
grep "smollm2_135m_5layer" spd/registry.py

# Verify YAML syntax
python3 -c "import yaml; yaml.safe_load(open('spd/experiments/lm/smollm2_135m_5layer_config.yaml'))"
```

### ☐ 2. Create Code Archive
```bash
git archive --format=tar.gz --output=/tmp/sjcs-spd.tar.gz HEAD
```

### ☐ 3. Verify Credentials
```bash
# Check .env exists locally (for reference)
cat /Users/seane/iCloudDrive/Documents/GitHub/sjcs-spdenv
```

**Required credentials:**
- WANDB_API_KEY
- WANDB_ENTITY
- HF_TOKEN (optional, but recommended)

---

## POD DEPLOYMENT (RunPod Console)

### ☐ 4. Navigate to GPU Cloud
- Go to: https://www.runpod.io/console/gpu-cloud
- Login if needed

### ☐ 5. Filter for Target Configuration
**Filters to apply:**
- GPU Type: A100-SXM4-80GB
- GPU Count: 4
- Cloud Type: Secure Cloud
- Storage: 100GB minimum

### ☐ 6. Verify Pod Specifications BEFORE Deploying
**CRITICAL - Check these exact values:**
- [ ] GPU Model shows: "A100-SXM4-80GB" (NOT "A100-PCIE")
- [ ] GPU Count: 4
- [ ] Price/hr: $6.96 (±$0.20 acceptable)
- [ ] Cloud type: "Secure Cloud" (NOT "Community Cloud")
- [ ] Storage: ≥100GB

### ☐ 7. Deploy Pod
- Click "Deploy"
- Wait for pod to start (usually 1-3 minutes)
- **SAVE THE FOLLOWING:**
  - Pod ID: ________________
  - Pod Name: ________________
  - SSH IP: ________________
  - SSH Port: ________________

---

## POD VERIFICATION (SSH Into Pod)

### ☐ 8. SSH Into Pod
```bash
ssh -p <PORT> root@<POD_IP>
```

### ☐ 9. CRITICAL: Verify NVLink Topology
```bash
nvidia-smi topo -m
```

**MUST SEE:**
- GPU0-GPU1: **NV12** or **NV4** (or similar NV#)
- GPU0-GPU2: **NV12** or **NV4**
- GPU0-GPU3: **NV12** or **NV4**
- GPU1-GPU2: **NV12** or **NV4**
- GPU1-GPU3: **NV12** or **NV4**
- GPU2-GPU3: **NV12** or **NV4**

**REJECT IF YOU SEE:**
- PIX, PXB, SYS, or PCIe (these indicate PCIe-only, NO NVLink)

**⚠️ IF NVLINK NOT FOUND:** STOP POD IMMEDIATELY, REQUEST REFUND, FIND NEW POD

### ☐ 10. Verify GPU Hardware
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

**Expected output:**
```
name, memory.total [MiB], driver_version
A100-SXM4-80GB, 81920 MiB, <driver_version>
A100-SXM4-80GB, 81920 MiB, <driver_version>
A100-SXM4-80GB, 81920 MiB, <driver_version>
A100-SXM4-80GB, 81920 MiB, <driver_version>
```

### ☐ 11. Check CUDA and PyTorch
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

**Expected:**
- PyTorch: ≥2.0.0
- CUDA: ≥11.8
- GPUs: 4

---

## ENVIRONMENT SETUP (On Pod)

### ☐ 12. Transfer Code Archive
```bash
# On local machine (new terminal)
scp -P <PORT> /tmp/sjcs-spd.tar.gz root@<POD_IP>:/workspace/
```

### ☐ 13. Extract Code
```bash
# On pod
cd /workspace
tar xzf sjcs-spd.tar.gz
rm sjcs-spd.tar.gz
ls -la  # Verify files extracted
```

### ☐ 14. Install System Dependencies
```bash
apt-get update
apt-get install -y git vim htop  # Optional: vim, htop for monitoring
```

### ☐ 15. Install Python Dependencies
```bash
# Core ML libraries (might already be installed)
python3 -m pip install --break-system-packages \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# SPD dependencies
python3 -m pip install --break-system-packages \
  transformers datasets wandb pydantic einops jaxtyping

# Install SPD package
cd /workspace
python3 -m pip install --break-system-packages -e .
```

### ☐ 16. Setup Credentials
```bash
cd /workspace
cat > .env << 'EOF'
WANDB_API_KEY=<your-wandb-api-key>
WANDB_ENTITY=SJCS-SPD
HF_TOKEN=<your-huggingface-token>
EOF

source .env
```

**Note**: Replace placeholders with actual credentials from `/Users/seane/iCloudDrive/Documents/GitHub/sjcs-spdenv`

---

## PRE-FLIGHT VALIDATION (On Pod)

### ☐ 17. Test Imports
```bash
python3 << 'EOF'
import spd
print("✓ SPD package")

import transformers
print("✓ Transformers")

import wandb
print("✓ WandB")

import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ GPU count: {torch.cuda.device_count()}")
EOF
```

### ☐ 18. Validate Config File
```bash
python3 << 'EOF'
import yaml

with open('spd/experiments/lm/smollm2_135m_5layer_config.yaml') as f:
    cfg = yaml.safe_load(f)

modules = len([p for p in cfg['target_module_patterns']])
print(f"✓ Target modules: {modules} (expected: 15)")

print(f"✓ Steps: {cfg['steps']} (expected: 30000)")
print(f"✓ Batch size: {cfg['batch_size']} (expected: 16)")
print(f"✓ Gradient accum: {cfg['gradient_accumulation_steps']} (expected: 4)")
print(f"✓ C components: {cfg['C']} (expected: 1000)")

assert modules == 15, "Wrong number of modules!"
assert cfg['steps'] == 30000, "Wrong number of steps!"
print("\n✓ CONFIG VALIDATION PASSED")
EOF
```

### ☐ 19. Test Model Loading (Quick Check)
```bash
python3 << 'EOF'
from transformers import AutoModel, AutoTokenizer

print("Loading SmolLM2-135M-Instruct...")
model = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

print(f"✓ Model loaded: {model.config.num_hidden_layers} layers")
print(f"✓ Vocab size: {model.config.vocab_size}")
print(f"✓ Hidden size: {model.config.hidden_size}")
EOF
```

---

## TRAINING LAUNCH (On Pod)

### ☐ 20. Create Launch Script
```bash
cat > /workspace/run_training.sh << 'EOF'
#!/bin/bash
set -e

# Load environment
cd /workspace
source .env

# Export for distributed training
export WANDB_START_METHOD=thread
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Clear Python cache
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

echo "========================================="
echo "Starting SmolLM2-135M 5-Layer SPD Training"
echo "========================================="
echo "Config: spd/experiments/lm/smollm2_135m_5layer_config.yaml"
echo "GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "========================================="

# Launch training
torchrun --nproc_per_node=4 --nnodes=1 --master_port=29500 \
    spd/experiments/lm/lm_decomposition.py \
    spd/experiments/lm/smollm2_135m_5layer_config.yaml \
    2>&1 | tee training.log

echo "========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "========================================="
EOF

chmod +x /workspace/run_training.sh
```

### ☐ 21. Record Start Time and Begin Training
```bash
# Record start time
date > /workspace/training_start_time.txt

# Launch in background with nohup (so it continues if SSH disconnects)
nohup /workspace/run_training.sh > /workspace/training_output.log 2>&1 &

# Save PID
echo $! > /workspace/training.pid
```

### ☐ 22. Monitor Initial Output (First 10 Steps)
```bash
# Watch log in real-time
tail -f /workspace/training_output.log
```

**Watch for:**
- [ ] All 4 ranks initialize (RANK 0, 1, 2, 3)
- [ ] "NCCL communication WORKS" message
- [ ] DDP model wrapped successfully
- [ ] WandB run created (look for wandb.ai URL)
- [ ] First step completes (note step time)
- [ ] No CUDA OOM errors
- [ ] No NCCL errors

**Expected first step time: 2.3-2.5 seconds**

---

## MONITORING (First Hour)

### ☐ 23. Monitor Every 10 Minutes (First Hour)
```bash
# Check if training is running
ps -p $(cat /workspace/training.pid) && echo "✓ Training running" || echo "✗ Training DIED"

# Check latest logs
tail -n 50 /workspace/training.log

# Check GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# Check step progress
grep "Step.*Loss" /workspace/training.log | tail -5
```

**Expected values:**
- GPU utilization: >80%
- Memory per GPU: ~10-20GB / 80GB
- Step time: 2.3-2.5s/step (consistent)

### ☐ 24. Calculate Cost Projection (After 100 Steps)
```bash
# Extract average step time from first 100 steps
python3 << 'EOF'
import re

with open('/workspace/training.log') as f:
    times = []
    for line in f:
        m = re.search(r'(\d+\.\d+)s/step', line)
        if m:
            times.append(float(m.group(1)))

if len(times) >= 10:
    avg_time = sum(times[-50:]) / len(times[-50:])  # Last 50 steps
    total_hours = (30000 * avg_time) / 3600
    total_cost = total_hours * 6.96

    print(f"Average step time: {avg_time:.2f}s")
    print(f"Projected total hours: {total_hours:.1f}h")
    print(f"Projected total cost: ${total_cost:.2f}")

    if total_cost > 160:
        print("\n⚠️  WARNING: Cost projection exceeds budget!")
EOF
```

**If cost > $160:** Consider aborting and investigating

---

## LONG-TERM MONITORING

### ☐ 25. Check Every 2-4 Hours
```bash
# Quick status check
echo "=== Training Status ==="
ps -p $(cat /workspace/training.pid) && echo "✓ Running" || echo "✗ Stopped"
tail -n 10 /workspace/training.log
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
```

### ☐ 26. Monitor WandB Dashboard
- Go to: https://wandb.ai/SJCS-SPD/smollm2-spd
- Check:
  - [ ] Run is active
  - [ ] Loss is decreasing
  - [ ] Metrics updating regularly
  - [ ] No error messages

### ☐ 27. Check Checkpoints
```bash
# List saved checkpoints
ls -lah /workspace/outputs/*/checkpoints/
```

**Expected:** Checkpoint every 5,000 steps

---

## FAILURE RECOVERY

### If Training Crashes:
```bash
# Check last error
tail -n 100 /workspace/training.log | grep -i error

# Check if checkpoint exists
ls /workspace/outputs/*/checkpoints/step_*.pt

# Resume from checkpoint (if available)
# - Update config to set resume_from_checkpoint path
# - Relaunch training
```

### If Cost Exceeds Budget:
- Abort training: `kill $(cat /workspace/training.pid)`
- Save outputs: `tar czf /workspace/outputs_backup.tar.gz /workspace/outputs/`
- Download results before stopping pod

### If Pod Crashes:
- Outputs should be in WandB
- Check WandB for last uploaded checkpoint
- Restart on new pod if needed

---

## COMPLETION

### ☐ 28. Verify Training Completed
```bash
grep "Training completed" /workspace/training_output.log
```

### ☐ 29. Calculate Final Cost
```bash
# Get end time
date > /workspace/training_end_time.txt

# Calculate duration
python3 << 'EOF'
from datetime import datetime

with open('/workspace/training_start_time.txt') as f:
    start = datetime.strptime(f.read().strip(), '%a %b %d %H:%M:%S %Z %Y')
with open('/workspace/training_end_time.txt') as f:
    end = datetime.strptime(f.read().strip(), '%a %b %d %H:%M:%S %Z %Y')

duration = (end - start).total_seconds() / 3600
cost = duration * 6.96

print(f"Duration: {duration:.2f} hours")
print(f"Total cost: ${cost:.2f}")
EOF
```

### ☐ 30. Verify WandB Upload
- Go to WandB run page
- Check final metrics uploaded
- Verify checkpoints available
- Download important artifacts

### ☐ 31. Stop Pod
- Go to RunPod console
- Stop pod to prevent further charges
- Delete pod after confirming all data saved

---

## SUCCESS CRITERIA

- ✅ Training completed all 30,000 steps
- ✅ Total cost < $150
- ✅ Final WandB run shows converged loss
- ✅ Checkpoints saved and accessible
- ✅ No catastrophic failures
- ✅ Average step time 2.3-2.5s

**DEPLOYMENT DATE:** __________
**DEPLOYED BY:** __________
**FINAL COST:** $__________
