#!/bin/bash

ORG_DIR=$PWD
PROGRESS_FILE="$PWD/worker_progress.stat"

chmod a+x common/*.sh

# create progress stat file
STAGE=$(<$PROGRESS_FILE)

cd common

if ! [ -e $PROGRESS_FILE ]; then
	./00-prepare-nodes.sh
	echo "0" > $PROGRESS_FILE
	STAGE=0
fi

# installing CUDNN and Nvidia-driver
if (( $STAGE < 1 )); then
	./01-install-cudnn-and-nvidia-driver.sh
	echo "1" > $PROGRESS_FILE
	sudo reboot
fi

# installing docker
if (( $STAGE < 2 )); then
	./02-install-docker.sh
	echo "2" > $PROGRESS_FILE
	sudo reboot
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
	sudo reboot
fi

cd $ORG_DIR
