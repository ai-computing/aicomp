#!/bin/bash

#docker run --rm -it --network host --gpus=all -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox --mount type=bind,src=/var/nfs,dst=/data --security-opt seccomp=seccomp-docker.json swsok/gpt-neox:v8
docker run --rm -it --network host --gpus=all -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox --mount type=bind,src=./dataset,dst=/data --security-opt seccomp=seccomp-docker.json swsok/gpt-neox:v8
