#!/bin/bash
#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Generate ProfileDB for all TP configurations using Docker
# For each TP config, sweeps MBS (1,2,4,8,...) until OOM
#
# Usage:
#   ./make_profiledb.sh <MODEL_SIZE> [MASTER_ADDR] [NODE1_HOSTNAME] [REMOTE_DIR]
#
# Example:
#   ./make_profiledb.sh 70B                                    # Single node, TP=1,2,4
#   ./make_profiledb.sh 70B 10.0.0.1 node1 /home/user/fasop   # Multi-node, TP=1,2,4,8
# node1 means the node that has node rank 1 with this script

# ============================================================================
# Configuration
# ============================================================================
CONTAINER_NAME="make_profiledb"
CONTAINER_IMAGE="make_profiledb:latest"
CONTAINER_WORKSPACE_DIR="/workspace/fasop"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")/docker"

# MBS values to try (2^N, stops on OOM)
MICRO_BATCH_SIZES=(1 2 4 8 16 32 64 128)

set -e

# ============================================================================
# Parse arguments
# ============================================================================
MODEL_SIZE=${1:-"70B"}
MASTER_ADDR=${2:-"127.0.0.1"}
NODE1_HOSTNAME=${3:-""}
REMOTE_DIR=${4:-"$(dirname "$SCRIPT_DIR")"}

# ============================================================================
# HuggingFace Token (required for Llama models)
# ============================================================================
if [ -z "$LLAMA_ACCESS_TOKEN" ]; then
    echo "ERROR: LLAMA_ACCESS_TOKEN not set"
    echo "Usage: LLAMA_ACCESS_TOKEN=hf_xxxxx ./make_profiledb.sh 70B ..."
    exit 1
fi
echo "===> LLAMA_ACCESS_TOKEN found (length: ${#LLAMA_ACCESS_TOKEN})"

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
            echo "Supported: 1B, 3B, 70B or full model name"
            exit 1
        fi
        ;;
esac

MODEL_FILENAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)

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

run_mbs_sweep() {
    local PP=$1
    local TP=$2
    local DP=$3
    local NNODES=$4
    local NODE_RANK=${5:-0}

    echo ""
    echo "================================================="
    echo " MBS Sweep: PP=$PP, TP=$TP, DP=$DP (Nodes=$NNODES)"
    echo " MBS to try: ${MICRO_BATCH_SIZES[*]}"
    echo "================================================="

    local SUCCESS_COUNT=0
    local OOM_REACHED=false

    for MBS in "${MICRO_BATCH_SIZES[@]}"; do
        BATCH_SIZE=$((MBS * 8))
        RUN_ID="${MODEL_FILENAME}-bs${BATCH_SIZE}-mbs${MBS}-pp${PP}-tp${TP}-dp${DP}"

        echo ""
        echo "-------------------------------------------------"
        echo " RUN_ID: $RUN_ID"
        echo " Batch=$BATCH_SIZE, MicroBatch=$MBS"
        echo "-------------------------------------------------"

        SECONDS=0
        docker exec $CONTAINER_NAME \
            /bin/bash -c "cd /workspace/fasop/make_profiledb && \
            mkdir -p results && \
            bash ./run_distributed.sh \
                \"$MODEL_NAME\" \
                \"$NODE_RANK\" \
                \"$MASTER_ADDR\" \
                \"$NNODES\" \
                8 \
                True \
                \"$PP\" \
                \"$TP\" \
                \"$DP\" \
                \"$BATCH_SIZE\" \
                \"$MBS\" \
                \"$RUN_ID\" \
            2>&1 | tee \"results/${RUN_ID}.log\"" || true

        DOCKER_EXIT=$?
        ELAPSED=$SECONDS

        # Read exit code from file
        EXIT_CODE=$DOCKER_EXIT
        EXIT_LOG="make_profiledb/tmp/exitcode_${RUN_ID}.txt"
        EXIT_LINE=$(docker exec $CONTAINER_NAME cat "$EXIT_LOG" 2>/dev/null || echo "$DOCKER_EXIT")
        if [[ "$EXIT_LINE" == *,* ]]; then
            EXIT_CODE="${EXIT_LINE%%,*}"
        else
            EXIT_CODE="$EXIT_LINE"
        fi

        STATUS=$(status_from_exit "$EXIT_CODE")

        # Record to CSV
        docker exec $CONTAINER_NAME bash -c "echo '$MODEL_SHORT,$BATCH_SIZE,$MBS,$PP,$TP,$DP,$STATUS,$ELAPSED' >> $RESULT_CSV"

        echo ">>> MBS=$MBS: $STATUS (exit=$EXIT_CODE, elapsed=${ELAPSED}s)"

        if [ "$EXIT_CODE" -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        elif [ "$EXIT_CODE" -eq 10 ]; then
            echo ">>> OOM detected - stopping MBS sweep for this config"
            OOM_REACHED=true
            break
        else
            echo ">>> ERROR ($STATUS) - stopping MBS sweep for this config"
            break
        fi

        sleep 3
    done

    echo ">>> Config PP=$PP,TP=$TP,DP=$DP: $SUCCESS_COUNT successful, OOM=$OOM_REACHED"
}

# ============================================================================
# 1. Setup Docker Container
# ============================================================================
echo "=============================================="
echo " ProfileDB Generation - All TP Configurations"
echo "=============================================="
echo " Model: $MODEL_NAME ($MODEL_SHORT)"
echo " Master: $MASTER_ADDR"
echo " MBS sweep: ${MICRO_BATCH_SIZES[*]}"
echo "=============================================="

echo ""
echo "===> Removing existing container '$CONTAINER_NAME'..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true
sleep 2

echo "===> Checking port range 29500-29509..."
for port in $(seq 29500 29509); do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "===> Cleaning up port $port"
        kill -9 $(lsof -t -i :$port) 2>/dev/null || true
        sleep 1
    fi
done

# Build image if needed
if ! docker image inspect $CONTAINER_IMAGE >/dev/null 2>&1; then
    echo "===> Building image '$CONTAINER_IMAGE'..."
    docker build --network=host -t $CONTAINER_IMAGE "$DOCKER_DIR"
fi

# Start container
echo "===> Starting container '$CONTAINER_NAME'..."
docker run -d --gpus all --name $CONTAINER_NAME \
    -v ${HOME}/aicomp/fasop:$CONTAINER_WORKSPACE_DIR \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --network=host \
    -w $CONTAINER_WORKSPACE_DIR \
    -e LLAMA_ACCESS_TOKEN="$LLAMA_ACCESS_TOKEN" \
    --entrypoint /bin/bash \
    $CONTAINER_IMAGE -c "tail -f /dev/null"

echo "===> Container ready."

# ============================================================================
# 2. Initialize Results CSV
# ============================================================================
RESULT_DIR="make_profiledb/results"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
RESULT_CSV="${RESULT_DIR}/${MODEL_FILENAME}_all_tp_${TIMESTAMP}.csv"

docker exec $CONTAINER_NAME bash -c "mkdir -p $RESULT_DIR && echo 'model,batch_size,micro_batch_size,pp,tp,dp,status,elapsed_sec' > $RESULT_CSV"

# ============================================================================
# 3. Run All TP Configurations
# ============================================================================

# Configuration table:
# TP | PP | DP | GPU/node | Nodes
#  1 |  8 |  1 |    8     |   1
#  2 |  4 |  1 |    8     |   1
#  4 |  2 |  1 |    8     |   1
#  8 |  2 |  1 |    8     |   2  (multi-node)

# only tp=8 test
echo ""
echo "[1/4] TP=1, PP=8, DP=1 (single node)"
echo "=============================================="
run_mbs_sweep 8 1 1 1

echo ""
echo "[2/4] TP=2, PP=4, DP=1 (single node)"
echo "=============================================="
run_mbs_sweep 4 2 1 1

echo ""
echo "[3/4] TP=4, PP=2, DP=1 (single node)"
echo "=============================================="
run_mbs_sweep 2 4 1 1

echo ""
echo "[4/4] TP=8, PP=2, DP=1 (multi-node)"
echo "=============================================="
if [ -z "$NODE1_HOSTNAME" ]; then
    echo "WARNING: NODE1_HOSTNAME not provided. Skipping TP=8."
    echo "To enable: ./make_profiledb.sh $MODEL_SIZE $MASTER_ADDR <NODE1_HOSTNAME> [REMOTE_DIR]"
else
    echo "Starting node1 via SSH: $NODE1_HOSTNAME"

    # Build image on node1 if not exists
    echo "===> Checking/building Docker image on $NODE1_HOSTNAME..."
    ssh "$NODE1_HOSTNAME" "if ! docker image inspect $CONTAINER_IMAGE >/dev/null 2>&1; then \
        echo '===> Building image on node1...'; \
        cd $REMOTE_DIR/docker && docker build --network=host -t $CONTAINER_IMAGE .; \
    else \
        echo '===> Image exists on node1'; \
    fi"

    # Start node1 container
    echo "===> Starting container on $NODE1_HOSTNAME..."
    ssh "$NODE1_HOSTNAME" "docker rm -f $CONTAINER_NAME 2>/dev/null || true && \
        docker run -d --gpus all --name $CONTAINER_NAME \
            -v \${HOME}/aicomp/fasop:$CONTAINER_WORKSPACE_DIR \
            -v \${HOME}/.cache/huggingface:/root/.cache/huggingface \
            --ipc=host --network=host \
            -w $CONTAINER_WORKSPACE_DIR \
            -e LLAMA_ACCESS_TOKEN=\"$LLAMA_ACCESS_TOKEN\" \
            --entrypoint /bin/bash \
            $CONTAINER_IMAGE -c 'tail -f /dev/null'"

    sleep 3  # Wait for node1 container

    # Run MBS sweep on both nodes for TP=8
    for MBS in "${MICRO_BATCH_SIZES[@]}"; do
        BATCH_SIZE=$((MBS * 8))
        RUN_ID="${MODEL_FILENAME}-bs${BATCH_SIZE}-mbs${MBS}-pp2-tp8-dp1"

        echo ""
        echo "-------------------------------------------------"
        echo " RUN_ID: $RUN_ID (multi-node)"
        echo "-------------------------------------------------"

        # Start node1 in background
        ssh "$NODE1_HOSTNAME" "docker exec $CONTAINER_NAME \
            /bin/bash -c \"cd /workspace/fasop/make_profiledb && \
            mkdir -p results && \
            bash ./run_distributed.sh \\\"$MODEL_NAME\\\" 1 \\\"$MASTER_ADDR\\\" 2 8 True 2 8 1 $BATCH_SIZE $MBS \\\"$RUN_ID\\\" \
            2>&1 | tee results/${RUN_ID}_node1.log\"" &
        NODE1_PID=$!

        # Run node0
        SECONDS=0
        docker exec $CONTAINER_NAME \
            /bin/bash -c "cd /workspace/fasop/make_profiledb && \
            mkdir -p results && \
            bash ./run_distributed.sh \"$MODEL_NAME\" 0 \"$MASTER_ADDR\" 2 8 True 2 8 1 $BATCH_SIZE $MBS \"$RUN_ID\" \
            2>&1 | tee \"results/${RUN_ID}_node0.log\"" || true

        DOCKER_EXIT=$?
        ELAPSED=$SECONDS

        wait $NODE1_PID 2>/dev/null || true

        EXIT_CODE=$DOCKER_EXIT
        EXIT_LOG="make_profiledb/tmp/exitcode_${RUN_ID}.txt"
        EXIT_LINE=$(docker exec $CONTAINER_NAME cat "$EXIT_LOG" 2>/dev/null || echo "$DOCKER_EXIT")
        [[ "$EXIT_LINE" == *,* ]] && EXIT_CODE="${EXIT_LINE%%,*}"

        STATUS=$(status_from_exit "$EXIT_CODE")
        docker exec $CONTAINER_NAME bash -c "echo '$MODEL_SHORT,$BATCH_SIZE,$MBS,2,8,1,$STATUS,$ELAPSED' >> $RESULT_CSV"

        echo ">>> MBS=$MBS: $STATUS"

        if [ "$EXIT_CODE" -eq 10 ]; then
            echo ">>> OOM - stopping TP=8 sweep"
            break
        elif [ "$EXIT_CODE" -ne 0 ]; then
            echo ">>> ERROR - stopping TP=8 sweep"
            break
        fi

        sleep 3
    done
fi

# ============================================================================
# 4. Summary
# ============================================================================
echo ""
echo "=============================================="
echo " ProfileDB Generation Complete!"
echo "=============================================="
echo " Results CSV: $RESULT_CSV"
echo " JSON files: make_profiledb/results/profile_*.json"
echo ""
echo " To convert to NPZ:"
echo "   python make_npz.py"
echo "=============================================="
