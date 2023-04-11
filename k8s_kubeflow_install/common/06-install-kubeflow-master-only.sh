#!/bin/bash

cd ~
git clone https://github.com/kubeflow/manifests.git
cd manifests
git checkout v1.6.0

kustomize build common/cert-manager/cert-manager/base | kubectl apply -f -
kubectl wait --for=condition=ready pod -l 'app in (cert-manager,webhook)' --timeout=180s -n cert-manager
kustomize build common/cert-manager/kubeflow-issuer/base | kubectl apply -f -

while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done

# wait until all pods become Running state
watch kubectl get pod -A

