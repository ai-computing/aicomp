# K8s and Kubeflow setup HOWTO
This document describes how to configure a Kubernetes cluster and Kubeflow environment.  
Kubernetes consists of master and worker nodes, and Kubeflow is the same.  
The following describes the common installation process for both master and worker nodes, and then describes the settings for master nodes and worker nodes respectably.  
All scripts and necessary files are placed in **common/** directory.  

## Common Setup
Change to common/ directory.
```
cd common
```
### 00. Preparing Nodes
Swap has to be disabled, because Kubernetes will throw an error if Swap is enabled on the nodes.
```
./00-prepare-nodes.sh
```
### 01. CUDNN and Nvidia driver
Remove existing nvidia drivers and cuda packages.  
Nouveau driver has to be disabled.  
Install nvidia-cuda-toolkit, cudnn, and nvidia-driver-525.
```
./01-install-cudnn-and-nvidia-driver.sh
sudo reboot
```
### 02. Docker
Kubernetes is based on Docker Engine.  
Install Docker and reboot.
```
./02-install-docker.sh
sudo reboot
```
### 03. Nvidia Container Toolkit
Install Nvidia Container Toolkit and run nvidia cuda docker to check validation.
```
./03-install-nvidia-docker.sh
```
### 04. Install Kubernetes
Install K8s v1.21.10-00.  
Newer versions of K8s have some errors with Kubeflow.
```
./04-install-k8s.sh
sudo reboot
```
## Master node
### 05. Configure Master of K8s
Setup Master node of K8s cluster.  
Create .kube/config and add local path as default storage.
```
./05-init-k8s-master-only.sh
sudo reboot
```
### 06. Install Kubeflow
Install Kubeflow from https://github.com/kubeflow.  
Build binaries and launch kubeflow pods.
```
./06-install-kubeflow-master-only.sh
```

## Worker node

...
