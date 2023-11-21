#!/bin/bash

#CONFIGS=(125M.yml 6-7B.yml)
CONFIG=125M.yml
CONT_NAME="gpt-neox-container"
BATCHS=(1 2 4 8 16 32 64 128)
#BATCHS=(32)
#GPUS=(1 2 3 4 5 6 7 8)
#gpus per node
GPUS=4
NODES=(s1 s8)
#Pipeline Parallel
#PP=(1)
PP=(1 2 4 8)
TRAIN_ITERS=200000
TARGET_LM_LOSS=0
TRAIN_TIME=1000
HOSTFILE=./scripts_swsok/hostfile

rm logs/*
rm checkpoints/* -rf
mkdir swsok-results/
mkdir checkpoints/

rm $HOSTFILE
for i in ${NODES[@]}; do
	echo "$i slots=$GPUS" >> $HOSTFILE
done

sed -i "/\"train_iters\"/c\   \"train_iters\": \\$TRAIN_ITERS," configs/$CONFIG
sed -i "/\"lr_decay_iters\"/c\   \"lr_decay_iters\": \\$TRAIN_ITERS," configs/$CONFIG
sed -i "/\"target_lm_loss\"/c\   \"target_lm_loss\": \\$TARGET_LM_LOSS," configs/$CONFIG
sed -i "/\"target_time_in_sec\"/c\   \"target_time_in_sec\": \\$TRAIN_TIME," configs/$CONFIG

for i in ${NODES[@]}; do
	ssh $i docker stop $CONT_NAME
	ssh $i docker run -d -it --name $CONT_NAME --rm --network host --gpus $GPUS -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox --mount type=bind,src=/var/nfs,dst=/var/nfs swsok/gpt-neox:v7
done

for b in ${BATCHS[@]}; do
	for p in ${PP[@]}; do
		echo "$CONFIG Nodes ${#NODES[@]} GPUS $GPUS BATCH $b Pipeline $p" > logs/current_test_setting.txt
			
		sed -i "/\"train_micro_batch_size_per_gpu\"/c\   \"train_micro_batch_size_per_gpu\": \\$b," configs/$CONFIG
		sed -i "/\"pipe_parallel_size\"/c\   \"pipe_parallel_size\": \\$p," configs/$CONFIG

		docker exec -it -w /gpt-neox $CONT_NAME ./deepy.py train.py configs/$CONFIG configs/local_setup.yml configs/etri_cluster.yml

		mv logs/*stdout.txt swsok-results/conf-$CONFIG-pp-$p-microbatch-$b-$(date '+%Y-%m-%d').txt
		rm logs/*
		rm checkpoints/* -rf

		sleep 1
	done
done

for i in ${NODES[@]}; do
	ssh $i docker stop $CONT_NAME
done


