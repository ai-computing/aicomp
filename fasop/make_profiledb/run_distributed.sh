#!/bin/bash
#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Distributed execution script for FX graph node-level profiling
#
# Usage:
#   ./run_distributed.sh MODEL_NAME NODE_RANK MASTER_ADDR NNODES NPROC USE_CACHE PP TP DP [BATCH_SIZE] [MICRO_BATCH_SIZE] [RUN_ID]
#
# Example:
#   ./run_distributed.sh meta-llama/Llama-3.3-70B-Instruct 0 127.0.0.1 1 8 True 8 1 1 8 1 run1
#

set -e

# Parse arguments
MODEL_NAME=${1:-"meta-llama/Llama-3.3-70B-Instruct"}
NODE_RANK=${2:-0}
MASTER_ADDR=${3:-"127.0.0.1"}
NNODES=${4:-1}
NPROC_PER_NODE=${5:-8}
USE_CACHE=${6:-"True"}
PP_SIZE=${7:-8}
TP_SIZE=${8:-1}
DP_SIZE=${9:-1}
BATCH_SIZE=${10:-2}
MICRO_BATCH_SIZE=${11:-1}
RUN_ID=${12:-"default"}

MASTER_PORT=${MASTER_PORT:-29500}

echo "=============================================="
echo " FX Graph Node-Level Profiling"
echo "=============================================="
echo " Model: $MODEL_NAME"
echo " Node: $NODE_RANK / $NNODES"
echo " Master: $MASTER_ADDR:$MASTER_PORT"
echo " GPUs per node: $NPROC_PER_NODE"
echo " PP=$PP_SIZE, TP=$TP_SIZE, DP=$DP_SIZE"
echo " Batch=$BATCH_SIZE, MicroBatch=$MICRO_BATCH_SIZE"
echo " RUN_ID: $RUN_ID"
echo "=============================================="

# Run with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    profile_fx.py \
    --model_name "$MODEL_NAME" \
    --pp_size $PP_SIZE \
    --tp_size $TP_SIZE \
    --dp_size $DP_SIZE \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --run_id "$RUN_ID" \
    --use_cache $USE_CACHE

echo "=============================================="
echo " Profiling completed!"
echo " Check results/ directory for output files"
echo "=============================================="
