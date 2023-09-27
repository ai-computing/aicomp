#!/bin/bash

docker run --rm -it --gpus=all -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox swsok/gpt-neox:v3
