#!/bin/bash

ORG_DIR=$PWD
PROGRESS_FILE="$PWD/progress.stat"

chmod a+x common/*.sh

# create progress stat file
if ! [ -e $PROGRESS_FILE ]; then
	echo "0" > $PROGRESS_FILE
fi

STAGE=$(<$PROGRESS_FILE)
echo "stage=$STAGE"

cd common

# installing CUDNN and Nvidia-driver
if (( $STAGE < 1 )); then
	./01-install-cudnn-and-nvidia-driver.sh
	echo "1" > $PROGRESS_FILE
fi

# installing docker
if (( $STAGE < 2 )); then
	./02-install-docker.sh
	echo "2" > $PROGRESS_FILE
fi

# installing nvidia docker - for testing docker and gpus
if (( $STAGE < 3 )); then
	./03-install-nvidia-docker.sh
	echo "3" > $PROGRESS_FILE
fi

# installing Kubernetes
if (( $STAGE < 4 )); then
	./04-install-k8s.sh
	echo "4" > $PROGRESS_FILE
fi

# configuring Kubernetes
if (( $STAGE < 5 )); then
	./05-init-k8s-master-only.sh
	echo "5" > $PROGRESS_FILE
fi

cd $ORG_DIR
