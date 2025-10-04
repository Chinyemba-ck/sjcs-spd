#!/bin/bash
# NOTE: Set WANDB_API_KEY, HF_TOKEN as environment variables before running
export WANDB_START_METHOD=thread
export WANDB_ENTITY=SJCS-SPD

# CRITICAL: NCCL_P2P_DISABLE=1 required for PCIe bridge topology (PXB, no NVLink)
# P2P enabled causes all_reduce deadlock on ranks 1-2
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_TIMEOUT=7200

source .venv/bin/activate

torchrun --nproc_per_node=3 --nnodes=1 --master_port=29500 \
    spd/experiments/lm/lm_decomposition.py \
    spd/experiments/lm/gemma_270m_multilingual_3layer_config.yaml
