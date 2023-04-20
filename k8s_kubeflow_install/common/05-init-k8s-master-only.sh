#!/bin/bash
# init k8s
sudo kubeadm init --pod-network-cidr=10.217.0.0/16

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config


kubectl cluster-info

# CNI
kubectl create -f https://raw.githubusercontent.com/cilium/cilium/v1.6/install/kubernetes/quick-install.yaml
kubectl get pods -n kube-system --selector=k8s-app=cilium

kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta6/nvidia-device-plugin.yml


#test GPU
kubectl -n kube-system get pod -l name=nvidia-device-plugin-ds
kubectl -n kube-system logs  -l name=nvidia-device-plugin-ds

# default storageclass
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
kubectl get storageclass
kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
kubectl get sc

# install kustomize 
# 
if [ ! -f /usr/local/bin/kusomize ]
  then
    echo "kustomize"
    wget https://github.com/kubernetes-sigs/kustomize/releases/download/v3.2.0/kustomize_3.2.0_linux_amd64
    mv ./kustomize_3.2.0_linux_amd64 kustomize
    sudo chmod 777 kustomize
    sudo mv kustomize /usr/local/bin/kustomize
fi


# autocomplete k8s
shellname=`echo $SHELL | rev | cut -d '/' -f1 | rev`
shellconf=`echo ~/.\${shellname}rc`
grep -n "kubectl completion" $shellconf

if [ $? = 1 ]
  then
    echo 'install autocomplete k8s'
    sudo apt-get install bash-completion -y
    echo 'source <(kubectl completion '$shellname')' >>$shellconf
    echo 'alias k=kubectl' >>$shellconf
    echo 'complete -F __start_kubectl k' >>$shellconf
fi
