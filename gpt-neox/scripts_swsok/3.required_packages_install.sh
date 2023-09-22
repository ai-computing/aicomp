#!/bin/bash

sudo apt install libopenmpi-dev
pip install mpi4py
# urllib3 v2 conflicts with original gpt-neox codes
#pip uninstall urllib3
#pip install urllib3==1.26.16
