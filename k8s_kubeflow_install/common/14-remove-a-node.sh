#!/bin/bash

if [ -z $1 ]; then
	echo "Usage: $0 [node name to remove]"
	exit 0
fi

kubectl drain $1 --delete-local-data --force --ignore-daemonsets 
kubectl delete node $1

ssh $1 "sudo kubeadm reset"
