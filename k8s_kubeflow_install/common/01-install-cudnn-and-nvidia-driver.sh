#!/bin/bash

sudo apt-get -y update
sudo apt-get -y remove --purge '^nvidia-.*'
sudo apt-get -y remove --purge 'cuda-.*'

sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo update-initramfs -u

sudo apt-get -y install nvidia-cuda-toolkit
nvcc -V
whereis cuda
#mkdir ~/nvidia
#cd ~/nvidia
CUDNN_DEB_FILE="cudnn-local-repo-ubuntu2004-8.8.0.121_1.0-1_amd64.deb"
if ! [ -e $CUDNN_DEB_FILE ]; then
	sudo apt-get -y install axel
	axel -n 20  https://developer.download.nvidia.com/compute/redist/cudnn/v8.8.0/local_installers/12.0/${CUDNN_DEB_FILE}
<<<<<<< HEAD
=======
#	wget  https://developer.download.nvidia.com/compute/redist/cudnn/v8.8.0/local_installers/12.0/${CUDNN_DEB_FILE}
>>>>>>> 1fa1c6b77722000453667cc07bfd33920d5e633e
fi
sudo dpkg -i ${CUDNN_DEB_FILE}
sudo cp /var/cudnn-local-repo-ubuntu2004-8.8.0.121/cudnn-local-A9E17745-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt -y install libcudnn8=8.8.0.121-1+cuda12.0
sudo apt -y install libcudnn8-dev=8.8.0.121-1+cuda12.0
sudo apt -y install libcudnn8-samples=8.8.0.121-1+cuda12.0

source ~/.bashrc

sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo apt install -y nvidia-driver-525-server

