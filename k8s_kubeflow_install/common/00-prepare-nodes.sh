#!/bin/bash

sudo apt update
sudo apt upgrade -y

#disable swap
sudo swapoff -a
sudo sed -e '/swap/ s/^#*/#/' -i /etc/fstab

nodelistfile='nodes.txt'
USER=etri-aicomputing
echo Start
while read p; do
#	sudo vi /etc/hosts
#	sudo vi /etc/vim/vimrc
	ssh-copy-id $USER@$p
#do manually
#	ssh $USER@$p sudo bash -c 'echo "etri-aicomputing ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers'
#	ssh $USER@$p sudo bash -c 'echo "//129.254.202.240/Vimo-S_Shared_Folder  /mnt/nas        cifs    vers=2.1,username=swsok,password=swsok3723,noperm       0       0" >> /etc/fstab'
#	ssh $USER@$p sudo mkdir /mnt/nas
#	ssh $USER@$p sudo apt update
#	ssh $USER@$p sudo apt install -y cifs-utils
#	ssh $USER@$p sudo mount /mnt/nas
#	cd /mnt/nas/swsok/k8s/k8s_kubeflow_install_scripts/k8s-kubeflow-gpu-install-scripts-by-swsok
done < "$nodelistfile"
