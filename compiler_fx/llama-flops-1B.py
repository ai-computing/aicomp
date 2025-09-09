#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# *** This program was tested with torch 2.5.0 and transformers 4.46.2.
#     The version of transformers used must be consistent across all machines used for testing ***
#


import torch
import torch.nn as nn


import torch.distributed as dist
import datetime
import logging
import os
import sys
import math
import time
from packaging import version

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal

from functools import partial
from ptflops import get_model_complexity_info

logging.basicConfig(level=logging.ERROR)


#
# This program needs 'access token' for Llama. First, obtain your access token for Llama !!!
#
if len(sys.argv) > 1:
    os.environ['LLAMA_ACCESS_TOKEN'] = sys.argv[1]

access_token = os.getenv('LLAMA_ACCESS_TOKEN')
if access_token is None:
    raise ValueError("LLAMA_ACCESS_TOKEN environment variable is not set."
                    "       [Usage:] python3 llama-flops-1B.py <llama_access_token>")


#
# This program needs torch version 2.3.1 or higher !!!
#
required_version = "2.3.1"
current_version = torch.__version__

if version.parse(current_version) >= version.parse(required_version):
    print(f"torch version 2.3.1 or higher --> OK")
else:
    print(f"current torch version is {current_version}.")
    raise ValueError('This program needs torch version 2.3.1 or higher.')

#
# This program needs transformers version 4.46.2 or higher !!!
#
required_tf_version = "4.46.2"
current_tf_version = transformers.__version__

if version.parse(current_tf_version) >= version.parse(required_tf_version):
    print(f"transformers version 4.46.2 or higher --> OK")
else:
    print(f" current transformers version is {current_tf_version}.")
    raise ValueError('This program needs transformers version 4.46.2 or higher.')


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=access_token, use_cache=False)

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


print ('Total parameters in model: {:,}'.format(get_total_params(model)))


def llama_input_constructor(input_shape, tokenizer):
    inp_seq = ""
    for _ in range(input_shape[1] - 2):  # [CLS], [SEP]
        inp_seq += tokenizer.pad_token

    inputs = tokenizer([inp_seq] * input_shape[0], padding=True, truncation=True, max_length=1024, return_tensors="pt")
    data, labels = inputs.input_ids, inputs.input_ids
    temp = {'input_ids': data , 'labels' : labels }
    return temp



macs, params = get_model_complexity_info(model, (1, 1024), as_strings=True, input_constructor=partial(llama_input_constructor, tokenizer=tokenizer), print_per_layer_stat=True, verbose=True)
print('>>>>>>> # of Operations:', macs, ', # of Parameters', params)

