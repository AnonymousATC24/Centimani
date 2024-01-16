#!/usr/bin/env python
# coding: utf-8

# # Mirco-Benchmarking for Transformers
# 

import torch

print('Pytorch version\t:', torch.__version__)
print('CUDA version\t:', torch.version.cuda)
print('GPU\t\t:',torch.cuda.get_device_name())


# Let's first define a `walltime` method to benchmark Pytorch statements by at least 3 seconds. 

# In[2]:


import inspect
from collections import defaultdict
import pandas as pd
from torch.utils import benchmark 

pd.options.display.precision = 3

def var_dict(*args):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return dict([(name, val) for name, val in callers_local_vars if val is arg][0] 
                for arg in args)

def walltime(stmt, arg_dict, duration=3):
    return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(
        min_run_time=duration).median


# Last install huggingface from source code.

# In[3]:


import os

os.system("git clone https://github.com/huggingface/transformers")
os.system("cd transformers; pip install .")


# ## BERT Layer
# 
# The main body of a Transformer model is a stacking of Transformer blocks. Let's benchmark the performance of a single block. In BERT, it is often called a BERT layer. Let's construct one such layer from the [BERT large model](https://huggingface.co/bert-large-uncased). We use 16-bit floating points for better performance. 

from transformers import AutoConfig, BertLayer

config = AutoConfig.from_pretrained("bert-large-uncased")
layer = BertLayer(config).bfloat16().cuda()


# Then define a function to benchmark both forward and forward with backward performance using different sequence lengths and batch sizes. 

def layer_benchmark(layer, hidden_size, seq_lens, batch_sizes, cross_attention=False):
    h = hidden_size
    results = defaultdict(lambda: {})    
    encoder_state = 'encoder_hidden_states=X' if cross_attention else ''
    for s in seq_lens:
        for b in batch_sizes:            
            ffn = 16*b*s*h*h / 1e12  # TFLOPS for the Feed-Forward Network
            atten = (4*b*h*s*s + 8*b*s*h*h) / 1e12  # TFLOPS for attention            
            forward = ffn + (2 if cross_attention else 1) * atten
            
            X = torch.randn(b, s, h).bfloat16().cuda()
            results[f'batch={b}'][f'fwd seq_len={s}'] = forward / walltime(
                f'layer(X, {encoder_state})', var_dict(layer, X))
            results[f'batch={b}'][f'bwd seq_len={s}'] = 3 * forward / walltime(
                f'layer(X, {encoder_state})[0].sum().backward()', var_dict(layer, X))            
    return pd.DataFrame(results)


# In BERT pre-training, we often train with a sequence of 128 (stage 1) or 512 (stage 2). Let's test its performance. 

df = layer_benchmark(layer, config.hidden_size, [512], [8, 16, 32, 64])

print(df.to_string())


# ## GPT-2 Block
# 
# Next let's evaluate `gpt2-medium`, which has a similar architecture has `bert-large`, i.e. 24 layers with a 1024 hidden size. GPT2 is trained with a 1024 sequence length.


from transformers.models.gpt2.modeling_gpt2 import GPT2Block

config = AutoConfig.from_pretrained("gpt2-medium")
layer = GPT2Block(config, layer_idx=0).bfloat16().cuda()
df = layer_benchmark(layer, config.n_embd, [1024], [8, 16, 32, 64])

print(df.to_string())


# ## Conclusion
