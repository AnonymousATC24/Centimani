# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path

import numpy as np
import popart

# Add benchmark module to path
bench_path = Path(Path(__file__).absolute().parent,
                  'utils')
sys.path.append(str(bench_path))
#print(f'\n\nbench_path: {bench_path}\n\n')
from benchmarks.popart.benchmark import Benchmark, parse_opts, run

SecsPerIteration = []
ItemsPerSec = []
Tflops = []

def kaiming_init(shape, fan_in, opts, a=5.0, b=3.0):
    # shape = [out_channel, in_channel, size, size]
    stddev = np.sqrt(a) / np.sqrt(fan_in)
    bound = np.sqrt(b) * stddev

    dType_str = opts.dtype

    # These are from argument choices.
    if dType_str.lower() == 'float16':
        dType = np.float16
    elif dType_str.lower() == 'float':
        dType = np.float32

    return np.random.uniform(-bound, bound, shape).astype(dType)


def graph_builder(opts):
    if opts.mode == 'infer':
        builder_fn = infer_builder
    elif opts.mode == 'eval':
        builder_fn = eval_builder
    elif opts.mode == 'train':
        builder_fn = train_builder
    else:
        raise ValueError("Unknown mode '{}'".format(opts.mode))
    defn = builder_fn(opts)
    defn[0] = defn[0].getModelProto()
    return defn


def infer_builder(opts):
    builder = popart.Builder()

    input_width = opts.input_width
    input_height = opts.input_height
    channel_size = opts.channel_size
    batch_size = opts.batch_size
    filter_number = opts.filter_number
    kernel_size = opts.kernel_size
    padding = opts.padding
    stride = opts.stride
    dType_str = opts.dtype

    # These are from argument choices.
    if dType_str.lower() == 'float16':
        dType = np.float16
    elif dType_str.lower() == 'float':
        dType = np.float32

    # input shape in NCHW format
    input_shape = [batch_size, channel_size, input_height, input_width]
    d1 = popart.TensorInfo(dType_str, input_shape)
    if opts.use_zero_values:
        input = np.zeros(input_shape, dType)
    else:
        input = np.random.uniform(-1, 1, input_shape).astype(dType)

    i1 = builder.addInputTensor(d1, "input_tensor")
    out = builder.aiOnnx.relu([i1], "relu1")
    builder.addOutputTensor(out)

    return [
        builder,
        {i1: input},
        {out: popart.AnchorReturnType("ALL")},
        None,
        None
    ]


def eval_builder(opts):
    builder, data, outputs, __, __ = infer_builder(opts)

    dType_str = opts.dtype

    probs = builder.aiOnnx.softmax([list(outputs)][0])
    output_height = (opts.input_height + 2*opts.padding - opts.kernel_size)//opts.stride + 1
    output_width = (opts.input_width + 2*opts.padding - opts.kernel_size)//opts.stride + 1
    output_shape = [opts.batch_size, opts.filter_number, output_height, output_width]
    label = builder.addInputTensor(popart.TensorInfo(dType_str, output_shape))
    # Sum of square error
    loss = builder.aiOnnx.sub([label, probs])
    loss = builder.aiOnnx.reducesumsquare([loss])

    if opts.use_zero_values:
        label_data = np.zeros(output_shape, np.int32)
    else:
        label_data = np.random.uniform(0, 2, output_shape).astype(np.int32)

    return [
        builder,
        {**data, label: label_data},
        {loss: popart.AnchorReturnType("ALL")},
        loss,
        None
    ]


def train_builder(opts):
    builder, data, outputs, loss, __ = eval_builder(opts)

    return [
        builder,
        data,
        outputs,
        loss,
        popart.ConstSGD(0.01)
    ]

# w = width (input-width)
# h = height (input-height)
# c = channels in (channel-size)
# n = batch size (batch-size)


def add_args(parser):
    parser.add_argument('--batch-size', type=int, default=300, #
                        help='Set batch size.')
    parser.set_defaults(batches_per_step=1, steps=10,
                        mode='infer', auto_sharding=True)
    parser.add_argument('--warmup', type=int, default=5, #
                        help='Number warm-up steps')
    parser.add_argument('--input-width', type=int, default=54, #
                        help='Input width size')
    parser.add_argument('--input-height', type=int, default=54, #
                        help='Input height size')
    parser.add_argument('--channel-size', type=int, default=32, #
                        help='Channel size')
    parser.add_argument('--filter-number', type=int, default=64,
                        help='Number of filters')
    parser.add_argument('--kernel-size', type=int, default=3,
                        help='Kernel size')
    parser.add_argument('--padding', type=int, default=3, #
                        help='Number of padding')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for convolution')
    parser.add_argument('--dtype', type=str, default='FLOAT16',
                        choices=['FLOAT16', 'FLOAT'],
                        help='Variable type')
    parser.add_argument('--kernel', type=int, default=1,
                        choices=[1, 2, 3, 4],
                        help='Kernel number')

    return parser


def iteration_report(opts, time):

    global SecsPerIteration
    global ItemsPerSec
    global Tflops

    itemsPerSec = opts.batch_size * opts.batches_per_step / time

    w = opts.input_width
    h = opts.input_height
    c = opts.channel_size
    n = opts.batch_size
    tflops = 2 * w * h * c * n
    tflops_forw = tflops/time/10**12

    SecsPerIteration.append(time)
    ItemsPerSec.append(itemsPerSec)
    Tflops.append(tflops_forw)

    return "{:5f} items/sec, {:10f} TFLOPS".format(itemsPerSec, tflops_forw)

def Average(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':
    module = Benchmark(
        graph_builder,
        add_args,
        iteration_report
    )

    opts = parse_opts(module)
    #print(f'opts: {opts}')

    # Log Benchmark Message
    print("\n\n\n\nPopART Convolutional layer {} Synthetic benchmark.\n"
          " Kernel num {}.\n"
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Steps {}.\n"
          " Input width {}.\n"
          " Input height {}.\n"
          " Channel {}.\n"
          .format(
              {"infer": "Inference", "eval": "Evaluation",
                  "train": "Training"}[opts.mode],
              opts.kernel,
              opts.batch_size,
              opts.batches_per_step,
              opts.steps,
              opts.input_width,
              opts.input_height,
              opts.channel_size))
    mode = {"infer": "Inference", "eval": "Evaluation", "train": "Training"}[opts.mode]
    print(f"\n\nPopART Convolutional layer {mode} Synthetic benchmark.\n")
    print("mfg,mode,dtype,sort,kernel,Batch_size,Batches_per_Step,Steps,Input_width,Input_height,Channel,Compile_Time,sec_per_iter,items_per_second,TFLOPS,end_to_end")
    mfg = 'Graphcore'
    if opts.dtype == 'FLOAT16':
        sort = 'F16'
    else:
        sort = 'F32'

    print(f'{mfg},{mode},{opts.dtype},{sort},Kernel {opts.kernel},{opts.batch_size},{opts.batches_per_step},{opts.steps},{opts.input_width},{opts.input_height},{opts.channel_size},')
    np.random.seed(42)
    run(module, opts)


    SecsPerIteration    = SecsPerIteration[opts.warmup:]
    ItemsPerSec         = ItemsPerSec[opts.warmup:]
    Tflops              = Tflops[opts.warmup:]
    print(f"{Average(SecsPerIteration)}    sec/itr.   {Average(ItemsPerSec)} items/sec,   {Average(Tflops)} TFLOPS")
