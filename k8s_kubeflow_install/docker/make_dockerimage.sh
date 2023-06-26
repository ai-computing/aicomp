#!/bin/bash


cp Dockerfile.scratch Dockerfile
docker build --no-cache -t swsok/nvidia-pytorch-kubeflow:v1 .
docker login --username swsok --password etri-aicomputing
docker push swsok/nvidia-pytorch-kubeflow:v1

#cp Dockerfile.org Dockerfile
#docker build --no-cache -t swsok/jupyter-pytorch-cuda-full-sudo:v1.5.0 .
#docker login --username swsok --password etri-aicomputing
#docker push swsok/jupyter-pytorch-cuda-full-sudo:v1.5.0
