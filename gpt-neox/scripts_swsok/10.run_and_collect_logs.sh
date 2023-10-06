#!/bin/bash

CONFIGS=(125M.yml 6-7B.yml)
CONT_NAME="gpt-neox-container"
BATCHS=(1 2 4 8 16 32 64 128)

docker stop $CONT_NAME
rm logs/*
rm checkpoints/* -rf

for conf in ${CONFIGS[@]}; do
	for i in {1..8}; do
		docker run -d -it --name $CONT_NAME --rm --gpus $i -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox swsok/gpt-neox:v3

		for b in ${BATCHS[@]}; do
			sed -i "/\"train_micro_batch_size_per_gpu\"/c\   \"train_micro_batch_size_per_gpu\": \\$b," configs/$conf
			docker exec -it -w /gpt-neox $CONT_NAME ./deepy.py train.py configs/$conf configs/local_setup.yml
			mv logs/*stdout.txt swsok-results/conf-$conf-gpunum-$i-microbatch-$b-$(date '+%Y-%m-%d').txt
			rm logs/*
			rm checkpoints/* -rf
		done

		docker stop $CONT_NAME
		sleep 1
	done
done
