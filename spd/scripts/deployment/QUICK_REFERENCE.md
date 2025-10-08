# SmolLM2 5-Layer SPD - Quick Reference

## Pod Configuration
- **GPUs:** 4× A100-SXM4-80GB
- **Cloud:** Secure (on-demand)
- **Price:** $6.96/hr
- **Storage:** 100GB

## Key Commands

### Verify NVLink (CRITICAL - Run First!)
```bash
nvidia-smi topo -m
# MUST see: NV12 or NV4 (NOT PIX/PXB/SYS)
```

### Setup Code
```bash
cd /workspace
tar xzf sjcs-spd.tar.gz
source .env
python3 -m pip install --break-system-packages -e .
```

### Launch Training
```bash
/workspace/run_training.sh
```

### Monitor Training
```bash
# Check if running
ps -p $(cat /workspace/training.pid)

# Watch logs
tail -f /workspace/training.log

# GPU status
nvidia-smi

# Cost projection
grep "Projected total cost" /workspace/training.log
```

## Expected Values
- **Step time:** 2.3-2.5 seconds
- **GPU util:** >80%
- **Memory:** ~10-20GB per GPU
- **Duration:** 19-21 hours
- **Cost:** $134-145

## Abort If:
- ❌ NVLink shows PIX/PXB/SYS
- ❌ Step time > 3.0s consistently
- ❌ Projected cost > $160
- ❌ CUDA OOM errors
- ❌ NCCL communication failures

## WandB
- **Project:** smollm2-spd
- **Run prefix:** smollm2_135m_5layer
- **URL:** https://wandb.ai/SJCS-SPD/smollm2-spd

## Files
- **Config:** `spd/experiments/lm/smollm2_135m_5layer_config.yaml`
- **Log:** `/workspace/training.log`
- **Checkpoints:** `/workspace/outputs/*/checkpoints/`

## Emergency
```bash
# Kill training
kill $(cat /workspace/training.pid)

# Save outputs
tar czf outputs_backup.tar.gz /workspace/outputs/

# Check last error
tail -100 /workspace/training.log | grep -i error
```
