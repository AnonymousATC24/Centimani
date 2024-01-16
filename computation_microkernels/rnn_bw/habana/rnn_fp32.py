import argparse
import torch
import torch.nn as nn
import time

steps = 1000 # nb of steps in loop to average perf
nDryRuns = 100 # nb of warmup steps

parser = argparse.ArgumentParser(description='PyTorch Convnet Benchmark')
parser.add_argument('--inference', action='store_false', default=False,
                    help='run inference only')
parser.add_argument('--time_step', type=int, default=50,
                    help='time step')
parser.add_argument('--input_size', type=int, default=500,
                    help='input size')
parser.add_argument('--hidden_size', type=int, default=500,
                    help='hidden size')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--layers', type=int, default=1,
                    help='number of layers');
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='use bidirectional rnn')
parser.add_argument('--model_type', type=str, default='lstm',
                    choices=['lstm', 'gru'],
                    help='Type of RNN models, Options are [lstm|gru]')

args = parser.parse_args()

device = torch.cuda.current_device()
#dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
dtype = torch.float32

### rnn parameters
L = args.layers
T = args.time_step
N = args.batch_size
I = args.input_size
H = args.hidden_size
D = 2 if args.bidirectional else 1


import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True
kernel_name = 'cudnn'

endtoend_start = time.time()

def _time():
    torch.cuda.synchronize()
    return time.time()

if args.model_type == 'gru':
    hx = torch.randn(L*D, N, H).type(dtype).to(device)
    rnn = nn.GRU
else:
    hx_, cx_ = torch.randn(L*D, N, H).type(dtype).to(device), torch.randn(L*D, N, H).type(dtype).to(device)
    hx = (hx_, cx_)
    rnn = nn.LSTM

x = torch.randn(T, N, I).type(dtype).to(device)
model = rnn(I, H, L, bidirectional=args.bidirectional, dtype = dtype).to(device)

if args.inference:
    model.eval()
else:
    model.train()

model.cuda()

for i in range(nDryRuns):
    y, _ = model(x, hx)
    if not args.inference:
        y.mean().backward()

time_fwd, time_bwd = 0, 0

for i in range(steps):
    t1 = _time()
    y, _ = model(x, hx)
    t2 = _time()
    time_fwd = time_fwd + (t2 - t1)
    if not args.inference:
        y.mean().backward()
        t3 = _time()
        time_bwd = time_bwd + (t3 - t2)

endtoend_end = time.time()

time_fwd_avg = time_fwd / steps
time_bwd_avg = time_bwd / steps
tflop = 2 * 4 * T * I * H * 1 * N + 2 * 4 * T * H * H * 1 * N
tflops_fwd = tflop / time_bwd_avg / 10**12

print("A100,Training,torch.float32,Kernel,1000,%d,%d,%d,%d,0.0,%f,%f,%f,%f" % (T, N, I, H, time_bwd_avg / N, N / time_bwd_avg, tflops_fwd, endtoend_end-endtoend_start))
