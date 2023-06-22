#!/bin/bash

nohup kubectl port-forward --address="0.0.0.0" svc/istio-ingressgateway -n istio-system 8080:443 &
