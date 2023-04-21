# K8s and Kubeflow setup HOWTO
This document describes how to configure a Kubernetes cluster and Kubeflow environment.  
Kubernetes consists of master and worker nodes, and Kubeflow is the same.  
The following describes the common installation process for both master and worker nodes, and then describes the settings for master nodes and worker nodes respectably.  
All scripts and necessary files are placed in **common/** directory.  
If you want to know more about kubeflow, check up  
https://github.com/myoh0623/kubeflow  
and  
https://www.youtube.com/watch?v=tDqatoU2fhM&list=PL6ZWs3MJaiphOwtHQvBCA4GNw-EPDely-&ab_channel=%ED%95%9C%EC%96%91%EB%82%A8%EC%9E%90.  
I borrowed some scripts from the git. Thanks.  

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
Wait until all pods are "Running" Status.
### 07. Add Certification
Map HTTPS port of Kubeflow Dashboard to gateway, and add HTTPS certification.
```
./07-certificate-kubeflow-master-only.sh
```

## Worker node
Repeat followings for each worker node.
### 09. Join a worker node
Print K8s join command in master node.
```
./09-print-join-cmd.sh
```
Then, the join command will be printed. (ex. sudo kubeadm join 192.168.0.27:6443 --token pis8vh.t37edt4hiu9ldsdh --discovery-token-ca-cert-hash sha256:f0acbe5893e5115907f4326c135876b2d0f748303e440730bcf9216d7593cb27)

In a worker node, type the command to join the K8s cluster.
```
sudo kubeadm join 192.168.0.27:6443 --token pis8vh.t37edt4hiu9ldsdh --discovery-token-ca-cert-hash sha256:f0acbe5893e5115907f4326c135876b2d0f748303e440730bcf9216d7593cb27
```

For another worker node, run ./09-print-join-cmd.sh again. Because the join command is only valid for one node.  

## Master node
Run port forward command.
```
08-port-forward-kubeflow-master-only.sh
```
Then you can connect Kubeflow dashboard through web browser using following url.  
```
https://[master node ip]:8080
```
You will get some warning. Ignore it.  

Default user's ID and Password
```
user@example.com
12341234
```
