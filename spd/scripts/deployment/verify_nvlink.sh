#!/bin/bash
# NVLink Verification Script for RunPod GPU Pods
# This script MUST be run immediately after pod deployment to verify NVLink connectivity

set -e

echo "================================="
echo "NVLink Verification Script"
echo "================================="
echo ""

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "✓ Detected $GPU_COUNT GPUs"
echo ""

# Check GPU models
echo "GPU Models:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# CRITICAL: Check NVLink topology
echo "================================="
echo "NVLINK TOPOLOGY CHECK (CRITICAL)"
echo "================================="
nvidia-smi topo -m

echo ""
echo "================================="
echo "VERIFICATION INSTRUCTIONS:"
echo "================================="
echo ""
echo "✓ GOOD (NVLink active): Look for 'NV4', 'NV12', 'NV18', or similar"
echo "✗ BAD  (PCIe only):     Look for 'PIX', 'PXB', 'SYS'"
echo ""
echo "Example GOOD output:"
echo "  GPU0-GPU1: NV12"
echo "  GPU0-GPU2: NV12"
echo ""
echo "Example BAD output:"
echo "  GPU0-GPU1: PIX"
echo "  GPU0-GPU2: SYS"
echo ""
echo "⚠️  IF YOU SEE PIX/PXB/SYS:"
echo "   1. STOP the pod immediately"
echo "   2. Request refund from RunPod"
echo "   3. Deploy a different pod"
echo "   4. DO NOT proceed with training"
echo ""
echo "================================="

# Check CUDA and PyTorch
echo ""
echo "Python/CUDA Check:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU count (PyTorch): {torch.cuda.device_count()}')"

echo ""
echo "================================="
echo "Verification complete!"
echo "================================="
