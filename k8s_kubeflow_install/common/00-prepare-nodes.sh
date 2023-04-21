#!/bin/bash

sudo apt update
sudo apt upgrade -y

#disable swap
sudo swapoff -a
sudo sed -e '/swap/ s/^#*/#/' -i /etc/fstab

nodelistfile='nodes.txt'
USER=etri-aicomputing

if [ -e $nodelistfile ]; then
	while read p; do
		ssh-copy-id $USER@$p
		#for passwordless sudo
		#ssh $USER@$p sudo bash -c 'echo "etri-aicomputing ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers'
	done < "$nodelistfile"
fi
