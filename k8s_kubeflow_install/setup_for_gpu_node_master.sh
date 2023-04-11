#!/bin/bash

PROGRESS_FILE="progress.stat"

chmod a+x common/*.sh

# create progress stat file
if ! [ -e $PROGRESS_FILE ]; then
	echo "0" > $PROGRESS_FILE
fi

STAGE=$(<$PROGRESS_FILE)
echo "stage=$STAGE"

# installing CUDNN and Nvidia-driver
if (( $STAGE < 1 )); then
	common/01-install-cudnn-and-nvidia-driver.sh
	echo "1" > $PROGRESS_FILE
fi

# installing docker
if (( $STAGE < 2 )); then
	common/02-install-docker.sh
	echo "2" > $PROGRESS_FILE
fi

# installing nvidia docker - for testing docker and gpus
if (( $STAGE < 3 )); then
	common/03-install-nvidia-docker.sh
	echo "3" > $PROGRESS_FILE
fi

# installing Kubernetes
if (( $STAGE < 4 )); then
	common/04-install-k8s.sh
	echo "4" > $PROGRESS_FILE
fi

# configuring Kubernetes
if (( $STAGE < 5 )); then
	cd common
	./05-init-k8s-master-only.sh
	cd ..
	echo "5" > $PROGRESS_FILE
fi


