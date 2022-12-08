#
# Copyright (c) 2022-present, ETRI, All rights reserved.
#


"""
Training Transformer models using Pipeline Parallelism
======================================================

** Original Author**: `Pritam Damania <https://github.com/pritamdamania87>`_

"""

#
# Transformer model on torchgpipe (GPU & microbatch version)
#
#   RUN_FLAG = True --> During program execution, Out Of Order technology is
#                        applied to this transformer model.
#                        Then, this model's nn.Linear is replaced with OutGradOnlyLinear.
#



import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from torch.nn import TransformerEncoder, TransformerEncoderLayer

###
from torchgpipe import GPipe
from torchgpipe.gpipe import verify_module
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
#import torch.nn as nn


###
torch.manual_seed(42)

if sys.platform == 'win32':
    print('Windows platform is not supported for pipeline parallelism')
    sys.exit(0)
if torch.cuda.device_count() < 2:
    print('Need at least two GPU devices for this tutorial')
    sys.exit(0)

###

set_devices_cnt = 0

class Hooking():
    def __init__(self, module, name):
        self.forwrad_hook = module.register_forward_hook(self.fwd_h)
        self.backward_hook = module.register_backward_hook(self.bwd_h)
        self.weight = module.weight
        self.name = name
        self.fwd_inputs = []
        self.grad_outputs = []
        self.device = 0
        self.main_grad = None
        #
        #self.n_fwd = 0
        #self.n_bwd = 0

    def fwd_h(self, module, input, output):
        self.fwd_inputs.append(input[0])
        ### DEBUG
        #self.n_fwd = self.n_fwd + 1
        #
        #if self.n_fwd < 10:
        #    print(f'fwd_h: input[0] ==> { input[0].size() }, n_fwd:{self.n_fwd}')

    def bwd_h(self, module, input, output):
        self.grad_outputs.append(output[0])
        ### DEBUG
        #self.n_bwd = self.n_bwd + 1
        #
        #if self.n_fwd < 10:
        #    print(f'bwd_h: output[0] ==> { output[0].size() }, n_bwd: {self.n_bwd}')

    def compute_weight_grad(self):
        # DEBUG
        with torch.cuda.device(self.device):
            grad_output = self.grad_outputs.pop(0)
            # DEBUG
            #total_input = self.fwd_inputs.pop(0)
            total_input = self.fwd_inputs.pop()

            ### DEBUG
            #if self.n_fwd < 10:
            #    print(f'grad_output ==> { grad_output.size() }')
            #    print(f'total_input ==> { total_input.size() }')

            #
            #grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
            grad_output = grad_output.reshape(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
            #
            #total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
            total_input = total_input.reshape(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
            #
            #print(f'grad_output 2 ==> { grad_output.size() }')
            #print(f'total_input 2 ==> { total_input.size() }')
            #if self.n_fwd < 10:
            #    print(f'grad_weight ==> { self.weight.size() }')
            grad_weight = grad_output.t().matmul(total_input)

            with torch.no_grad():
                if self.main_grad is None:
                    self.main_grad = grad_weight
                else:
                    self.main_grad.add_(grad_weight)
                self.weight.grad = None


    def check_consistency(self):
        size = len(self.grad_outputs)
        assert size == 0, f"grad output size is not zero. SIZE:{size}, name:{self.name}"

        size = len(self.fwd_inputs)
        assert size == 0, f"forward inputs size is not zero. SIZE:{size}, name:{self.name}"

    def set_device_for_weight_grad(self):
        global set_devices_cnt

        if self.device != self.weight.device:
            set_devices_cnt = set_devices_cnt + 1

        self.device = self.weight.device

_HOOKING_LIST = []

def compute_weight_grad_all(n_microbatches, n_parts):
    global _HOOKING_LIST

    for n in range(n_parts):
        for i in range(n_microbatches):
            for h in _HOOKING_LIST[n]:
                h.compute_weight_grad()

def check_memory(n_parts):
    global _HOOKING_LIST

    for n in range(n_parts):
        for h in _HOOKING_LIST[n]:
            h.check_consistency()

def set_devices_for_weight_grad_all(n_parts):
    global _HOOKING_LIST

    for n in range(n_parts):
        for h in _HOOKING_LIST[n]:
            h.set_device_for_weight_grad()

def adjust_weight_grad_all(n_parts):
    global _HOOKING_LIST

    for n in range(n_parts):
        for h in _HOOKING_LIST[n]:
            h.weight.grad = h.main_grad


class OutGradOnlyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, weight, bias):
        ctx.save_for_backward(data, weight)
        ctx.use_bias = bias is not None

        output = torch.matmul(data, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        grad_input = grad_output.matmul(weight)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, None, grad_bias

class OutGradOnlyLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OutGradOnlyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        output = OutGradOnlyMatMul.apply(x, self.weight, self.bias)
        return output

def replace_modules1(model, old_name, new_class):
    for name, module in model.named_modules():
        if module.__class__.__name__ == old_name:
            module.__class__ = new_class

def run_for_hooking(model, n_partitions):
    global _HOOKING_LIST

    n_part = 0
    n_element = 0

    for partition in model.partitions:
        _HOOKING_LIST.append([])
        modules = partition.named_modules()
        for name, module in modules:
            #print(f'module name = { module.__class__.__name__}')
            if module.__class__.__name__ == "OutGradOnlyLinear":
                h = Hooking(module, name)
                _HOOKING_LIST[n_part].append(h)
                n_element = n_element + 1
        n_part = n_part + 1

    print(f'hooking lists: {n_part}, total elements : {n_element}')

def print_modules(model):
    print("################################################")
    modules = model.named_modules()
    for name, module in modules:
        print(name, module)


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
        x = x + self.pe[:x.size(0), :]
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

device = torch.device("cuda")

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
###
nlayers = 12 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 16 # the number of heads in the multiheadattention models
###
dropout = 0.2 # the dropout value
#dropout = 0 # the dropout value

###
#from torch.distributed import rpc
#tmpfile = tempfile.NamedTemporaryFile()
#rpc.init_rpc(
#    name="worker",
#    rank=0,
#    world_size=1,
#    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
#        init_method="file://{}".format(tmpfile.name),
#        # Specifying _transports and _channels is a workaround and we no longer
#        # will have to specify _transports and _channels for PyTorch
#        # versions >= 1.8.1
#        _transports=["ibv", "uv"],
#        _channels=["cuda_ipc", "cuda_basic"],
#    )
#)

###
#num_gpus = 2
num_gpus = 4
partition_len = ((nlayers - 1) // num_gpus) + 1

# Add encoder in the beginning.
tmp_list = [Encoder(ntokens, emsize, dropout).cuda(0)]
module_list = []

# Add all the necessary transformer blocks.
for i in range(nlayers):
    transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
    if i != 0 and i % (partition_len) == 0:
        module_list.append(nn.Sequential(*tmp_list))
        tmp_list = []
    device = i // (partition_len)
    tmp_list.append(transformer_block.to(device))

# Add decoder in the end.
tmp_list.append(Decoder(ntokens, emsize).cuda(num_gpus - 1))
module_list.append(nn.Sequential(*tmp_list))

###
#from torch.distributed.pipeline.sync import Pipe

# Build the pipeline.
#chunks = 8
chunks = 4
#chunks = 1
###
#model = Pipe(torch.nn.Sequential(*module_list), chunks = chunks)

### when num_gpus = 2
#model = GPipe(torch.nn.Sequential(*module_list), chunks = chunks, balance=[1,1])
### when num_gpus = 4
#model = GPipe(torch.nn.Sequential(*module_list), chunks = chunks, balance=[1,1,1,1], 
#        devices=[0,1,2,3], checkpoint='never')
model = GPipe(torch.nn.Sequential(*module_list), chunks = chunks, balance=[1,1,1,1], 
        checkpoint='never')


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params

print ('Total parameters in model: {:,}'.format(get_total_params(model)))

###
print(model)
print("------------")
RUN_FLAG = True
#RUN_FLAG = False

n_partitions = len(model.devices)
n_chunks = model.chunks

if RUN_FLAG == True:
    replace_modules1(model, "Linear", OutGradOnlyLinear)
    print(model)
    run_for_hooking(model, n_partitions)

print(f'model len={len(model)}, partitions={n_partitions}, chunks={n_chunks}')


criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

###
in_device = model.devices[0]
out_device = model.devices[-1]

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
        # Since the Pipe is only within a single host and process the ``RRef``
        # returned by forward method is local to this node and can simply
        # retrieved via ``RRef.local_value()``.
        ###
        #output = model(data).local_value()
        ###
        data = data.to(in_device, non_blocking=True)
        targets = targets.to(out_device, non_blocking=True)
        output = model(data)
        # Need to move targets to the device where the output of the
        # pipeline resides.
        ###
        output = output.to(out_device, non_blocking=True)
        if i == 0 and RUN_FLAG == True:
            set_devices_for_weight_grad_all(n_partitions)
        #loss = criterion(output.view(-1, ntokens), targets.cuda(1))
        loss = criterion(output.view(-1, ntokens), targets.cuda(out_device))
        loss.backward()
        ###
        if RUN_FLAG == True:
            compute_weight_grad_all(n_chunks, n_partitions)
            adjust_weight_grad_all(n_partitions)

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

    ###
    if RUN_FLAG == True:
        check_memory(n_partitions)

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(vocab)
    # Evaluate only for 50 batches to keep script execution time low.
    nbatches = min(50 * bptt, data_source.size(0) - 1)
    with torch.no_grad():
        for i in range(0, nbatches, bptt):
            data, targets = get_batch(data_source, i)
            ###
            data = data.to(in_device, non_blocking=True)
            targets = targets.to(out_device, non_blocking=True)
            ###
            #output = eval_model(data).local_value()
            output = eval_model(data)
            ###
            output = output.to(out_device, non_blocking=True)
            output_flat = output.view(-1, ntokens)
            # Need to move targets to the device where the output of the
            # pipeline resides.
            ###
            #total_loss += len(data) * criterion(output_flat, targets.cuda(1)).item()
            total_loss += len(data) * criterion(output_flat, targets.cuda(out_device)).item()
    return total_loss / (len(data_source) - 1)

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

#
tick = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    ###
    #val_loss = evaluate(model, val_data)
    #print('-' * 89)
    #print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
    #      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
    #                                 val_loss, math.exp(val_loss)))
    #print('-' * 89)

    #if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()


###
#test_loss = evaluate(best_model, test_data)
#print('=' * 89)
#print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#    test_loss, math.exp(test_loss)))
#print('=' * 89)

tock = time.time()
print('#### Time elapsed: %.3f sec' % (tock - tick))


