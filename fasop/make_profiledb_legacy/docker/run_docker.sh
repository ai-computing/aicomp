#!/bin/bash

#docker run --gpus all -i -t --name cmh_2.3.1 --ipc=host --network=host swsok/fasop_docker:pytorch_2.5.0-cuda12.4-cudnn9-devel /bin/bash
sudo docker run --gpus all -i -t --name fasop_test_swsok --ipc=host --network=host swsok/fasop_docker:pytorch_2.5.0-cuda12.4-cudnn9-devel /bin/bash

