#!/bin/bash

CONFIGS=(6.7B-32k-len-conf.yml)
CONT_NAME="gpt-neox-container"
BATCHS=(1 2 4)
GPUS=8
#SEQLEN=(2048 4096 8192 16384 32768)
SEQLEN=(32768)
#SEQLEN=(32768)
TRAIN_ITERS=200
TARGET_LM_LOSS=0
TRAIN_TIME=1000
PP=(1 2 4)
MP=(1 2 4 8)
GRADACCSTEP=(16 32)

docker stop $CONT_NAME
rm logs/*
rm checkpoints/* -rf

i=$GPUS
#docker run -d -it --name $CONT_NAME --rm --gpus $GPUS -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox --mount type=bind,src=./dataset,dst=/data --security-opt seccomp=seccomp-docker.json swsok/gpt-neox:v8
docker run -d -it --name $CONT_NAME --rm --gpus $i -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox --security-opt seccomp=seccomp-docker.json swsok/gpt-neox:v8
for conf in ${CONFIGS[@]}; do

	sed -i "/\"train_iters\"/c\   \"train_iters\": \\$TRAIN_ITERS," configs/$conf
	sed -i "/\"lr_decay_iters\"/c\   \"lr_decay_iters\": \\$TRAIN_ITERS," configs/$conf
#sed -i "/\"target_lm_loss\"/c\   \"target_lm_loss\": \\$TARGET_LM_LOSS," configs/$conf
#sed -i "/\"target_time_in_sec\"/c\   \"target_time_in_sec\": \\$TRAIN_TIME," configs/$conf

	for p in ${PP[@]}; do
	sed -i "/\"pipe_parallel_size\"/c\   \"pipe_parallel_size\": \\$p," configs/$conf
	for m in ${MP[@]}; do
		sed -i "/\"model_parallel_size\"/c\   \"model_parallel_size\": \\$m," configs/$conf

		for b in ${BATCHS[@]}; do
			sed -i "/\"train_micro_batch_size_per_gpu\"/c\   \"train_micro_batch_size_per_gpu\": \\$b," configs/$conf

			for s in ${SEQLEN[@]}; do
				echo "$conf GPU $i microbatch $b pp $p mp $m" > logs/current_test_setting.txt
			
				sed -i "/\"seq_length\"/c\   \"seq_length\": \\$s," configs/$conf
				sed -i "/\"max_position_embeddings\"/c\   \"max_position_embeddings\": \\$s," configs/$conf
			for g in ${GRADACCSTEP[@]}; do
				sed -i "/\"gradient_accumulation_steps\"/c\   \"gradient_accumulation_steps\": \\$g," configs/$conf

				docker exec -it -w /gpt-neox $CONT_NAME ./deepy.py train.py configs/$conf configs/enwik8.yml
				mv logs/*stdout.txt swsok-results/conf-$conf-gpunum-$i-pp-$p-mp-$m-microbatch-$b-seqlen-$s-gradaccustep-$g-$(date '+%Y-%m-%d').txt
				rm logs/*
				rm checkpoints/* -rf
			done
			done
		done

		sleep 1
	done
done
done

docker stop $CONT_NAME
