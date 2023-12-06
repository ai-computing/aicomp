#!/bin/bash

#CONFIGS=(125M.yml 6-7B.yml)
CONFIG=20B.yml
CONT_NAME="gpt-neox-container"
#BATCHS=(1 2 4 8 16 32 64 128)
BATCHS=(1 2 4 8 16 32 64)
#GPUS=(1 2 3 4 5 6 7 8)
#gpus per node
GPUS=8
NODES=(s1 s8 s2 s3)
#NODES=(s1 s8)
#Pipeline Parallel
PP=(1 2 4 8 16 32)
MP=(1 2 4 8 16 32)
TRAIN_ITERS=4
TARGET_LM_LOSS=0
TRAIN_TIME=60000
HOSTFILE=./scripts_swsok/hostfile

if [ ! -z "$1" ]; then
	GPUS=$1
fi

rm logs/*
rm 20B_checkpoints/* -rf
mkdir swsok-results/
mkdir 20B_checkpoints/

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

for p in ${PP[@]}; do
	sed -i "/\"pipe_parallel_size\"/c\   \"pipe_parallel_size\": \\$p," configs/$CONFIG
	for m in ${MP[@]}; do
		sed -i "/\"model_parallel_size\"/c\   \"model_parallel_size\": \\$m," configs/$CONFIG
		for b in ${BATCHS[@]}; do
			echo "$CONFIG Nodes ${#NODES[@]} GPUS $GPUS BATCH $b Pipeline $p Modelparallel $m" > logs/current_test_setting.txt
			
			sed -i "/\"train_micro_batch_size_per_gpu\"/c\   \"train_micro_batch_size_per_gpu\": \\$b," configs/$CONFIG

			docker exec -it -w /gpt-neox $CONT_NAME ./deepy.py train.py configs/$CONFIG configs/etri_cluster.yml
			SAVELOG=swsok-results/conf-$CONFIG-nodes-${#NODES[@]}-gpus-$GPUS-pp-$p-mp-$m-microbatch-$b-$(date '+%Y-%m-%d').txt
			mv logs/*stdout.txt $SAVELOG
			rm logs/*
			rm 20B_checkpoints/* -rf

			LINE=`grep lm_loss $SAVELOG|wc -l`
			if [ "$LINE" = "0" ]; then
				echo "lm_loss not found in log file $SAVELOG. - deleted it"
				rm $SAVELOG
				break
			fi 
			sleep 1
		done
	done
done

for i in ${NODES[@]}; do
	ssh $i docker stop $CONT_NAME
done

