# fasop_test

How to build docker image for FASOP test.

```
docker build -t fasop_test:latest .
```
This repository doesn't have FASOP codes.(tdpp.zip)


Use swsok/fasop_docker:pytorch_2.5.0-cuda12.4-cudnn9-devel docker image from dockerhub.


It has all files to run FASOP test.





How to run FASOP db creation.
```
./run_fasop.sh [host ip] [huggingface token]
```
You need to create huggingface access token before running it.
