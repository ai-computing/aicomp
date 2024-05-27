#!/bin/bash

CONFIG=760M-32k-len-conf.yml
CONT_NAME="gpt-neox-container"
BATCHS=(1 2)
GPUS=8
#SEQLEN=(2048 4096 8192 16384 32768)
SEQLEN=(32768)
#SEQLEN=(32768)
TRAIN_ITERS=20
TARGET_LM_LOSS=0
TRAIN_TIME=1000
GRADACCSTEP=(32 64)

docker stop $CONT_NAME
sudo rm logs/* -rf
rm checkpoints/* -rf

i=$GPUS
conf=$CONFIG

#docker run -d -it --name $CONT_NAME --rm --gpus $GPUS -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox --mount type=bind,src=./dataset,dst=/data --security-opt seccomp=seccomp-docker.json swsok/gpt-neox:v8
docker run -d -it --name $CONT_NAME --rm --gpus $i -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox --security-opt seccomp=seccomp-docker.json swsok/gpt-neox:v8


sed -i "/\"train_iters\"/c\   \"train_iters\": \\$TRAIN_ITERS," configs/$conf
sed -i "/\"lr_decay_iters\"/c\   \"lr_decay_iters\": \\$TRAIN_ITERS," configs/$conf
#sed -i "/\"target_lm_loss\"/c\   \"target_lm_loss\": \\$TARGET_LM_LOSS," configs/$conf
#sed -i "/\"target_time_in_sec\"/c\   \"target_time_in_sec\": \\$TRAIN_TIME," configs/$conf

for b in ${BATCHS[@]}; do
	sed -i "/\"train_micro_batch_size_per_gpu\"/c\   \"train_micro_batch_size_per_gpu\": \\$b," configs/$conf

	for s in ${SEQLEN[@]}; do
		echo "$conf GPU $i microbatch $b pp $p mp $m" > logs/current_test_setting.txt
			
		sed -i "/\"seq_length\"/c\   \"seq_length\": \\$s," configs/$conf
		sed -i "/\"max_position_embeddings\"/c\   \"max_position_embeddings\": \\$s," configs/$conf

		for g in ${GRADACCSTEP[@]}; do
			sed -i "/\"gradient_accumulation_steps\"/c\   \"gradient_accumulation_steps\": \\$g," configs/$conf

			docker exec -it -w /gpt-neox $CONT_NAME ./deepy.py train.py configs/$conf configs/enwik8.yml
			sudo mv -f logs/*stdout.txt swsok-results/conf-$conf-gpunum-$i-zero-3-microbatch-$b-seqlen-$s-gradaccustep-$g-$(date '+%Y-%m-%d').txt
			sudo rm -rf logs/*
			rm checkpoints/* -rf
		done
	done

	sleep 1
done

docker stop $CONT_NAME
