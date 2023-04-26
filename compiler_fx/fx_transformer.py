#
#  FX based training applied to the Transformer model (CPU version)
#
#    This source is adapted from the pytorch tutorial (https://github.com/pytorch/tutorials/blob/main/advanced_source/ddp_pipeline.py)
#
# [prerequisite]
#   $ pip3 install torchtext
#   $ pip3 install torchdata


import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.optim import Adam
from torch import fx
from torch.fx.node import Node
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


#
torch.manual_seed(42)


class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)

class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).permute(1, 0, 2)


#
# slice wrapping function for FX's symbolic_tracing
#
def pe_slice(x, y):
    sizes = x.size(0)
    return y[:sizes, :]

torch.fx.wrap('pe_slice')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x = x + self.pe[:x.size(0), :]    # original

        #
        # wrap slice for FX's symbolic_tracing
        #
        x = x + pe_slice(x, self.pe)
        return self.dropout(x)



import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"]) 

def data_process(raw_text_iter):
  data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

#device = torch.device("cuda")
device = torch.device("cpu")

def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


bptt = 25
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    # Need batch dimension first for pipeline parallelism.
    return data.t(), target


ntokens = len(vocab) # the size of vocabulary
emsize = 4096 # embedding dimension
nhid = 4096 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 16 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

# Add encoder in the beginning.
tmp_list = [Encoder(ntokens, emsize, dropout).to(device)]
module_list = []

# Add all the necessary transformer blocks.
for i in range(nlayers):
    transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
    tmp_list.append(transformer_block.to(device))

# Add decoder in the end.
tmp_list.append(Decoder(ntokens, emsize).to(device))
module_list.append(nn.Sequential(*tmp_list))

model = torch.nn.Sequential(*module_list)

# LossWrapper: cited from PiPPy
class LossWrapper(torch.nn.Module):
    def __init__(self, module, loss_fn):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError("LossWrapper: no forward implementation")

# SimpleLossWrapper: cited from PiPPy
class SimpleLossWrapper(LossWrapper):
    def forward(self, x, targets):
        out1 = self.module(x)
        return self.loss_fn(out1, targets)


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

print ('Total parameters in model: {:,}'.format(get_total_params(model)))

criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


#
wrapper = SimpleLossWrapper(model, criterion) 

#
# FX's symbolic tracing
#
gm1 = fx.symbolic_trace(wrapper)

for node in gm1.graph.nodes:
    print(f"node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.args:{node.args}, node.all_input_nodes:{node.all_input_nodes}")

print("--------------------------------------------------------------------------")
print(gm1.code)
print("--------------------------------------------------------------------------")


#
def _get_loss_output(graph: fx.Graph):
     output_nodes = [n for n in graph.nodes if n.op == 'output']
     assert len(output_nodes) == 1
     output_node = output_nodes[0]
     loss_node = output_node.args[0]

     return loss_node, output_node

loss_node, output_node = _get_loss_output(gm1.graph)


# stage_backward function: cited from PiPPy
def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: List[int],
):

    #print(f" ** stage_backward - len(stage_output):{len(stage_output)}, len(output_grads):{len(output_grads)}, len(input_values):{len(input_values)}")

    stage_output_with_grads = [
        stage_output[i] for i in outputs_with_grads_idxs
    ]

    output_grads_with_grads = [
        output_grads[i] for i in outputs_with_grads_idxs
    ]

    stage_output_tensors = []
    output_grad_tensors = []

    def extract_tensors_with_grads(output_val, grad_val):
        if isinstance(output_val, torch.Tensor):
            if not output_val.requires_grad and output_val.grad_fn is None:
                return
            stage_output_tensors.append(output_val)
            output_grad_tensors.append(grad_val)
        elif isinstance(output_val, (tuple, list)):
            if grad_val is None:
                return
            for ov, gv in zip(output_val, grad_val):
                extract_tensors_with_grads(ov, gv)
        elif isinstance(output_val, dict):
            if grad_val is None:
                return
            for k in output_val.keys():
                extract_tensors_with_grads(output_val[k], grad_val[k])
        else:
            print(f"... ignored in this case")
            pass

    extract_tensors_with_grads(stage_output_with_grads, output_grads_with_grads)

    torch.autograd.backward(stage_output_tensors, grad_tensors=output_grad_tensors)

    grad_inputs = []
    for val in input_values:
        if isinstance(val, torch.Tensor):
            grad_inputs.append(val.grad)
        else:
            grad_inputs.append(None)

    barrier_token = None
    return grad_inputs, barrier_token


class FXRun:

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

        self.loss = None

        self.env: Dict[str, Node] = {}

        # TODO
        #self.fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}
        self.fwd_cache: Dict[str, Tuple[Any, List[torch.Tensor]]] = {}
        self.grads: Dict[str, Any] = {}



    def get_destination(self, input_nodes, lst_):
        for i, m in enumerate(input_nodes):
            for n in self.graph.nodes:
                if n.name == m.name:
                    #if m.op == 'call_module' or m.op == 'call_method':
                    if m.op == 'call_module' or m.op == 'call_method' or m.op == 'call_function':
                        lst_.append(m)
                        break

                    #if m.op == 'call_function':
                    #    self.get_destination(m.all_input_nodes, lst_)


    def fx_forward(self, *args):
        args_iter = iter(args)

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
                #print(f"placeholder: node.name:{node.name}, result:{result}")

            elif node.op == 'get_attr':
                target_atoms = node.target.split('.')
                attr_itr = self.mod
                for i , atom in enumerate(target_atoms):
                    if not hasattr(attr_itr, atom):
                        raise RuntimeError(\
                                f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                    attr_itr = getattr(attr_itr, atom)
                result = attr_itr

            elif node.op == 'call_function':
                #result = node.target(\
                #        *fx.graph.map_arg(node.args, lambda n: self.env[n.name]), \
                #        **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))

                flat_args = []
                def extract_tensor_args(b):
                    a = self.env[b.name]
                    nonlocal flat_args
                    if isinstance(a, torch.Tensor):
                        val = a.detach().requires_grad_(a.requires_grad)
                        flat_args.append(val)
                        return val
                    else:
                        flat_args.append(a)
                        return a
                
                    return a

                args = fx.graph.map_arg(node.args, extract_tensor_args)
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)
                
                result = node.target(*args, **kwargs)
                
                self.fwd_cache[node.name] = \
                        ( result if isinstance(result, tuple) else (result,), \
                        flat_args, )


            elif node.op == 'call_method':

                #self_obj, *args = fx.graph.map_arg(node.args, lambda n: self.env[n.name])
                #kwargs = fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name])
                #result = getattr(self_obj, node.target)(*args, **kwargs)

                arg0_b = node.args[0]

                arg0_a = self.env[arg0_b.name]

                #self_obj = arg0_a.detach().requires_grad_(arg0_a.requires_grad)
                if isinstance(arg0_a, torch.Tensor):
                    self_obj = arg0_a.detach().requires_grad_(arg0_a.requires_grad)
                else:
                    self_obj = arg0_a


                flat_args = [self_obj, ]

                def extract_tensor_args(b):
                    a = self.env[b.name]
                    nonlocal flat_args
                    if isinstance(a, torch.Tensor):
                        val = a.detach().requires_grad_(a.requires_grad)
                        flat_args.append(val)
                        return val
                    else:
                        flat_args.append(a)
                        return a

                    return a

                args = fx.graph.map_arg(node.args[1:], extract_tensor_args)
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)

                result = getattr(self_obj, node.target)(*args, **kwargs)

                self.fwd_cache[node.name] = \
                        ( result if isinstance(result, tuple) else (result,), \
                        flat_args, )

            elif node.op == 'call_module':
                #result = self.modules[node.target](\
                #        *fx.graph.map_arg(node.args, lambda n: self.env[n.name]),\
                #        **fx.graph.map_arg(node.kwargs, lambda n: self.env[n.name]))

                flat_args = []
                def extract_tensor_args(b):
                    a = self.env[b.name]
                    nonlocal flat_args
                    if isinstance(a, torch.Tensor):
                        val = a.detach().requires_grad_(a.requires_grad)
                        flat_args.append(val)
                        return val
                    else:
                        flat_args.append(a)
                        return a

                    return a

                args = fx.graph.map_arg(node.args, extract_tensor_args)
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)

                target_atoms = node.target.split('.')
                attr_itr = self.mod
                for i , atom in enumerate(target_atoms):
                    if not hasattr(attr_itr, atom):
                        raise RuntimeError(\
                                f"Node referenced nonexistant target{'.'.join(target_atoms[:i])}")
                    attr_itr = getattr(attr_itr, atom)
                submod = attr_itr

                #result = submod(*args, **kwargs)
                if node.target == 'loss_fn':
                    myargs = [None, None]
                    #print(f"#### args:{args}, kwargs:{kwargs}")
                    #print(f"#### args:{len(args)}, kwargs:{len(kwargs)}")
                    #print(f"#### args[0]:{args[0]}, args[1]:{args[1]}")
                    myargs[0] = args[0].reshape(-1, ntokens)
                    myargs[1] = args[1]
                    myargs = tuple(myargs)

                    result = submod(*myargs, **kwargs)
                else:
                    result = submod(*args, **kwargs)


                if node.target == 'loss_fn':
                    if not str(node.all_input_nodes[0]).startswith("target"):
                        self.output = self.env[str(node.all_input_nodes[0])]
                    self.grads[node.name] = (None,)

                self.fwd_cache[node.name] = \
                        ( result if isinstance(result, tuple) else (result,), \
                        flat_args, )

                if node.target == 'loss_fn':
                    self.loss = result

            self.env[node.name] = result

            # 'output' excluded when LossWrapper applied

        #return fx.graph.map_arg(self.env[node.name], lambda n: self.env[n.name])
        return self.output

    def fx_backward(self, *args):
        loss = args

        for node in reversed(self.graph.nodes):
            if node.op == 'output':
                pass

            #if node.op == 'call_module' or node.op == 'call_method':
            if node.op == 'call_module' or node.op == 'call_method' or node.op == 'call_function':


                def extract_tensor_args(b):
                    a = self.env[b.name]
                    if isinstance(a, torch.Tensor):
                        val = a.detach().requires_grad_(a.requires_grad)
                        return val
                    else:
                        return a

                args = ()
                kwargs = fx.graph.map_arg(node.kwargs, extract_tensor_args)

                kwargs = dict(kwargs)

                k1, k2 = self.fwd_cache.pop(node.name)

                kwargs["stage_output"] = k1
                kwargs["input_values"] = k2
                kwargs["output_grads"] = self.grads[node.name]
                kwargs["outputs_with_grads_idxs"] = [0]

                if node.target != 'loss_fn' and self.grads[node.name] == (None,):
                    continue

                result = stage_backward(*args, **kwargs)

                next_ = []
                self.get_destination(node.all_input_nodes, next_)

                cnt = len(result[0])

                for i, m in enumerate(next_):
                    if cnt > 1:
                        if isinstance(result[0][i], list) and result[0][i] != None:
                            self.grads[m.name] = torch.stack(result[0][i], 0)
                        else:
                            self.grads[m.name] = ((result[0][i], ) if not isinstance(result[0][i], tuple) else result[0][i])

                    else:
                        if isinstance(result[0], list) and result[0][0] != None:
                            self.grads[m.name] = torch.stack(result[0], 0)
                        else:
                            self.grads[m.name] = ((result[0], ) if not isinstance(result[0], tuple) else result[0])


for node in gm1.graph.nodes:
    print(f"node.op:{node.op}, node.name:{node.name}, node.target:{node.target}")

print("-------------")

fx_run = FXRun(gm1)


import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocab)

    # Train only for 50 batches to keep script execution time low.
    nbatches = min(50 * bptt, train_data.size(0) - 1)


    for batch, i in enumerate(range(0, nbatches, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()

        # 
        output1 = fx_run.fx_forward(data, targets)
        loss = fx_run.loss
        fx_run.fx_backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 10
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, nbatches // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


best_val_loss = float("inf")
epochs = 5 # The number of epochs
best_model = None

tick = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    train()

    scheduler.step()

tock = time.time()
print('#### Time elapsed: %.3f sec' % (tock - tick))
