#!/bin/bash
#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Docker-based FX Graph Node-Level Profiling Script
#
# Usage:
#   ./run_docker.sh <MODEL_SIZE> <NODE_RANK> <MASTER_ADDR> <NNODES> <NPROC> <USE_CACHE> <PP> <TP> <DP>
#
#   MODEL_SIZE: 1B, 3B, 70B or full model name (e.g., meta-llama/Llama-3.2-1B)
#
# Environment:
#   LLAMA_ACCESS_TOKEN: HuggingFace token for Llama models (required if USE_CACHE=False)
#                       Falls back to HF_TOKEN if LLAMA_ACCESS_TOKEN is not set
#
# Examples:
#   export LLAMA_ACCESS_TOKEN="hf_xxxxx"
#   ./run_docker.sh 70B 0 127.0.0.1 1 8 True 8 1 1
#   ./run_docker.sh 1B 0 127.0.0.1 1 2 True 2 1 1
#

CONTAINER_NAME="make_profiledb"
CONTAINER_IMAGE="make_profiledb:latest"
CONTAINER_WORKSPACE_DIR="/workspace/fasop"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Micro Batch Sizes to try (will stop on OOM)
# ============================================================================
MICRO_BATCH_SIZES=(1 2 4 8 16 32 64 128)

# ============================================================================
# Helper Functions
# ============================================================================
status_from_exit() {
  case "$1" in
    0)  echo "SUCCESS" ;;
    10) echo "OOM" ;;
    20) echo "DIST_ERROR" ;;
    30) echo "EXCEPTION" ;;
    40) echo "PEER_FAILED" ;;
    50) echo "TIMEOUT" ;;
    *)  echo "FAIL($1)" ;;
  esac
}

# ============================================================================
# 1. Remove existing container
# ============================================================================
echo "===> Removing '$CONTAINER_NAME' container."
docker rm -f $CONTAINER_NAME 2>/dev/null || true
sleep 2
echo "===> Container removed."

echo "===> Creating new container '$CONTAINER_NAME'."

# ============================================================================
# 2. Clean up ports
# ============================================================================
echo "===> Checking port range 29500-29509"
for port in $(seq 29500 29509); do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "===> Cleaning up port $port"
        kill -9 $(lsof -t -i :$port) 2>/dev/null || true
        sleep 1
    fi
done

# ============================================================================
# 3. Build image if not exists
# ============================================================================
if ! docker image inspect $CONTAINER_IMAGE >/dev/null 2>&1; then
    echo "===> Image '$CONTAINER_IMAGE' not found. Building..."
    if ! docker build --network=host -t $CONTAINER_IMAGE "$SCRIPT_DIR"; then
        echo "ERROR: Failed to build image. Exiting."
        exit 1
    fi
    echo "===> Image built."
else
    echo "===> Image '$CONTAINER_IMAGE' found."
fi

# ============================================================================
# 4. Run container
# ============================================================================
docker run -d --gpus all --name $CONTAINER_NAME \
    -v ${HOME}/aicomp/fasop:$CONTAINER_WORKSPACE_DIR \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --network=host \
    -w $CONTAINER_WORKSPACE_DIR \
    -e LLAMA_ACCESS_TOKEN=${LLAMA_ACCESS_TOKEN:-$HF_TOKEN} \
    --entrypoint /bin/bash \
    $CONTAINER_IMAGE -c "tail -f /dev/null"
echo "===> Container '$CONTAINER_NAME' created."

echo "===> Network interfaces:"
docker exec $CONTAINER_NAME hostname -I

# ============================================================================
# 5. Parse arguments
# ============================================================================
MODEL_SIZE=$1
NODE_RANK=$2
MASTER_ADDR=$3
NNODES=$4
NPROC_PER_NODE=$5
USE_CACHE=$6
PP_SIZE=$7
TP_SIZE=$8
DP_SIZE=$9

# Set MODEL_NAME based on MODEL_SIZE
case "$MODEL_SIZE" in
    70|70B|70b)
        MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
        MODEL_SHORT="70b"
        ;;
    1|1B|1b)
        MODEL_NAME="meta-llama/Llama-3.2-1B"
        MODEL_SHORT="1b"
        ;;
    3|3B|3b)
        MODEL_NAME="meta-llama/Llama-3.2-3B"
        MODEL_SHORT="3b"
        ;;
    *)
        if [[ "$MODEL_SIZE" == *"/"* ]]; then
            MODEL_NAME="$MODEL_SIZE"
            MODEL_SHORT=$(echo "$MODEL_SIZE" | grep -oE '[0-9]+[Bb]' | tr '[:upper:]' '[:lower:]')
            [ -z "$MODEL_SHORT" ] && MODEL_SHORT="unknown"
        else
            echo "Unknown model size: $MODEL_SIZE"
            echo "Supported: 1B, 3B, 70B or full model name (e.g., meta-llama/Llama-3.2-1B)"
            exit 1
        fi
        ;;
esac

MODEL_FILENAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)

echo ""
echo "================================================="
echo " Mode: profile (MBS sweep until OOM)"
echo " Model: $MODEL_NAME"
echo " PP=$PP_SIZE, TP=$TP_SIZE, DP=$DP_SIZE"
echo " MBS to try: ${MICRO_BATCH_SIZES[*]}"
echo "================================================="

# ============================================================================
# 6. Profile loop - increase MBS until OOM
# ============================================================================
RESULT_DIR="make_profiledb/results"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
RESULT_CSV="${RESULT_DIR}/${MODEL_FILENAME}_${TIMESTAMP}.csv"

# Create CSV header
docker exec $CONTAINER_NAME bash -c "mkdir -p $RESULT_DIR && echo 'model,batch_size,micro_batch_size,pp,tp,dp,status,elapsed_sec' > $RESULT_CSV"

SUCCESS_COUNT=0
OOM_REACHED=false

for MBS in "${MICRO_BATCH_SIZES[@]}"; do
    # batch_size = micro_batch_size * num_micro_batches (use MBS * PP for reasonable batch)
    BATCH_SIZE=$((MBS * 8))

    RUN_ID="${MODEL_FILENAME}-${BATCH_SIZE}-${MBS}-${PP_SIZE}-${TP_SIZE}-${DP_SIZE}"

    echo ""
    echo "================================================="
    echo " RUN_ID: $RUN_ID"
    echo " Batch=$BATCH_SIZE, MicroBatch=$MBS"
    echo " PP=$PP_SIZE, TP=$TP_SIZE, DP=$DP_SIZE"
    echo "================================================="

    LOGFILE="results/${RUN_ID}.log"

    # Run profiling
    SECONDS=0
    docker exec $CONTAINER_NAME \
        /bin/bash -c "cd /workspace/fasop/make_profiledb && \
        mkdir -p results && \
        bash ./run_distributed.sh \
            '$MODEL_NAME' \
            '$NODE_RANK' \
            '$MASTER_ADDR' \
            '$NNODES' \
            '$NPROC_PER_NODE' \
            '$USE_CACHE' \
            '$PP_SIZE' \
            '$TP_SIZE' \
            '$DP_SIZE' \
            '$BATCH_SIZE' \
            '$MBS' \
            '$RUN_ID' \
        2>&1 | tee '$LOGFILE'"

    DOCKER_EXIT=$?
    ELAPSED=$SECONDS

    # Read exit code from file (profile_fx.py saves it)
    EXIT_CODE=$DOCKER_EXIT
    EXIT_LOG="make_profiledb/tmp/exitcode_${RUN_ID}.txt"

    EXIT_LINE=$(docker exec $CONTAINER_NAME cat "$EXIT_LOG" 2>/dev/null || echo "$DOCKER_EXIT")
    if [[ "$EXIT_LINE" == *,* ]]; then
        EXIT_CODE="${EXIT_LINE%%,*}"
        ELAPSED="${EXIT_LINE##*,}"
    else
        EXIT_CODE="$EXIT_LINE"
    fi

    STATUS=$(status_from_exit "$EXIT_CODE")

    # Record result
    docker exec $CONTAINER_NAME bash -c "echo '$MODEL_SHORT,$BATCH_SIZE,$MBS,$PP_SIZE,$TP_SIZE,$DP_SIZE,$STATUS,$ELAPSED' >> $RESULT_CSV"

    echo ""
    echo ">>> MBS=$MBS: $STATUS (exit=$EXIT_CODE, elapsed=${ELAPSED}s)"

    if [ "$EXIT_CODE" -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo ">>> SUCCESS - continuing to next MBS..."
    elif [ "$EXIT_CODE" -eq 10 ]; then
        echo ">>> OOM detected - stopping MBS sweep"
        OOM_REACHED=true
        break
    else
        echo ">>> ERROR ($STATUS) - stopping MBS sweep"
        break
    fi

    # Cleanup between runs
    sleep 3
done

echo ""
echo "================================================="
echo " MBS Sweep Completed"
echo " Successful runs: $SUCCESS_COUNT"
echo " OOM reached: $OOM_REACHED"
echo "================================================="

# ============================================================================
# 7. Summary
# ============================================================================
echo ""
echo "================================================="
echo " All Completed!"
echo " Results CSV: $RESULT_CSV"
echo " JSON files: ${RESULT_DIR}/profile_*.json"
echo " NPZ files: make_profiledb/results/llama${MODEL_SHORT}_*.npz (auto-generated by profile_fx.py)"
echo "================================================="
