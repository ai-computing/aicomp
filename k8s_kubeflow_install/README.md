# K8s and Kubeflow setup HOWTO
This document describes how to configure a Kubernetes cluster and Kubeflow environment.  
Kubernetes consists of master and worker nodes, and Kubeflow is the same.  
The following describes the common installation process for both master and worker nodes, and then describes the settings for master nodes and worker nodes respectably.  
All scripts and necessary files are placed in common/ directory.  

## Common Setup
### 00. Preparing Nodes
Swap has to be disabled, because Kubernetes will throw an error if Swap is enabled on the nodes.
```
cd common
./00-prepare-nodes.sh
```

## Master node
...
...

## Worker node

...
