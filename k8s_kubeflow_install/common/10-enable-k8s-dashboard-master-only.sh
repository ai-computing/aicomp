#!/bin/bash

#kubectl apply -f dashboard-adminuser.yaml

#kubectl apply -f cluster-role-binding.yaml

#kubectl -n kubernetes-dashboard describe secrets
# copy admin-user's token value for kubernetes-dashboard

kubectl create serviceaccount admin-user
kubectl create clusterrolebinding test-user-binding --clusterrole=cluster-admin --serviceaccount=default:admin-user
kubectl get secrets
#specify admin-user's token name in next cmd
kubectl describe secret admin-user-token-8bjnr
