#!/bin/bash

#common args
DOCKER_IMG=swsok/nccl_tests:cuda12.9
CONTAINER_NAME="nccl_tests_docker"
COMMON_ARG="--name $CONTAINER_NAME --ipc=host --network=host"
LOG=nccl_tests_result.txt
GPU_PAIR="0,4"

docker rm -f $CONTAINER_NAME 2>/dev/null

docker run -d $COMMON_ARG --gpus '"device='"$GPU_PAIR"'"' $DOCKER_IMG sleep infinity

EXEC_LIST=(all_gather_perf all_reduce_perf alltoall_perf broadcast_perf gather_perf hypercube_perf reduce_perf reduce_scatter_perf scatter_perf sendrecv_perf)

echo "    unit=GB/s [out-of-place_bw] [bus_bw] [in-place_bw] [bus_bw]" | tee -a $LOG

for perf_test in "${EXEC_LIST[@]}"; do
	echo -n "$perf_test " | tee -a $LOG
	docker exec $CONTAINER_NAME ./build/$perf_test -b 1G -e 1G -f 2 -g 2 | grep float | awk '{print $7,$8,$11,$12}' | tee -a $LOG
#	docker exec $CONTAINER_NAME ./build/$perf_test -b 8 -e 1G -f 2 -g 2
done

docker stop $CONTAINER_NAME >/dev/null
docker rm $CONTAINER_NAME >/dev/null

