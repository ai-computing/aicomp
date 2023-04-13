#!/bin/bash

cmd=$(kubeadm token create --print-join-command)
echo "sudo $cmd"
