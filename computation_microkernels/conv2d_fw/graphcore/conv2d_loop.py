# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path

import numpy as np
import popart

# Add benchmark module to path
bench_path = Path(Path(__file__).absolute().parent,
                  'utils')
sys.path.append(str(bench_path))
from benchmarks.popart.benchmark import Benchmark, parse_opts, run

SecsPerIteration = []
ItemsPerSec = []
Tflops = []
g_tflops_forw       = 0
g_items_per_second  = 0
g_iterations        = 0


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
        d2 = np.zeros([filter_number, channel_size, kernel_size, kernel_size], dType)
        d3 = np.zeros([filter_number], dType)
        input = np.zeros(input_shape, dType)
    else:
        d2 = kaiming_init([filter_number, channel_size, kernel_size, kernel_size],
                          channel_size*input_height*input_width,
                          opts)
        d3 = kaiming_init([filter_number], channel_size*input_height*input_width, opts)
        input = np.random.uniform(-1, 1, input_shape).astype(dType)

    builder = popart.Builder()
    builder.setGraphName("main_graph")

    i1 = builder.addInputTensor(d1, "input_tensor")
    i2 = builder.addInitializedInputTensor(d2, "weights")
    # i3 = builder.addInitializedInputTensor(d3, "bias")

    M = builder.aiOnnx.constant(np.array(opts.loop_steps).astype(np.int64), "M")
    cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond")

    # loop body subgraph
    loop_builder = builder.createSubgraphBuilder()
    loop_builder.setGraphName("body")

    # loop body inputs: [iteration_number, condition_in, lcd_tensors]
    loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
    keepgoing = loop_builder.addInputTensor(
        popart.TensorInfo("BOOL", []))
    a_in = loop_builder.addUntypedInputTensor(i1)

    # a_out = loop_builder.aiOnnx.conv([a_in, i2, i3], strides=[stride, stride],
    #                           pads = [padding, padding, padding, padding])
    a_out = loop_builder.aiOnnx.conv([a_in, i2], strides=[stride, stride],
                              pads = [padding, padding, padding, padding])

    # loop body outputs: [condition_out, a_out]
    loop_builder.addOutputTensor(keepgoing)
    loop_builder.addOutputTensor(a_out)

    # Inputs: [iteration_number, condition_in, a_in]
    out = builder.aiOnnx.loop([M, cond, i1], 1, loop_builder)[0]

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

    input_width = opts.input_width
    input_height = opts.input_height
    #channel_size = opts.channel_size
    batch_size = opts.batch_size
    filter_number = opts.filter_number
    kernel_size = opts.kernel_size
    padding = opts.padding
    stride = opts.stride
    dType_str = opts.dtype




    output_height = (input_height + 2*padding - kernel_size)//stride + 1
    output_width = (input_width + 2*padding - kernel_size)//stride + 1
    output_shape = [batch_size, filter_number, output_height, output_width]





    M = builder.aiOnnx.constant(np.array(opts.loop_steps).astype(np.int64), "M")
    cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond")

    # loop body subgraph
    loop_builder = builder.createSubgraphBuilder()
    loop_builder.setGraphName("body")

    loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
    keepgoing = loop_builder.addInputTensor(
        popart.TensorInfo("BOOL", []))
    #...

    #a_out = loop_builder.aiOnnx.conv([a_in, i2], strides=[stride, stride],
    #                          pads = [padding, padding, padding, padding])

    # loop body outputs: [condition_out, a_out]
    loop_builder.addOutputTensor(keepgoing)
    loop_builder.addOutputTensor(a_out)

    # Inputs: [iteration_number, condition_in, a_in]
    out = builder.aiOnnx.loop([M, cond, outputs], 1, loop_builder)[0]

    builder.addOutputTensor(out)




    probs = builder.aiOnnx.softmax([list(out)][0])

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

# w         = width             = (input-width)
# h         = height            = (input-height)
# c         = channels in       = (channel-size)
# n         = batch size        = (batch-size)
# k         = kernel output channels    = (filter-number)
# s         = filter width      = (kernel-size)
# r         = filter height     = (kernel-size)
# pad_w     = padding width     = (padding)
# pad_h     = padding height    = (padding)
# wstride   = stride width      = (stride)
# hstride   = stride height     = (stride)


def add_args(parser):
    parser.add_argument('--loop-steps', type=int, default=128, 
                        help='Set loop steps')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Set batch size')
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
    global g_tflops_forw
    global g_items_per_second
    global g_iterations

    g_iterations += 1

    #itemsPerSec = opts.batch_size * opts.batches_per_step / time


    w_0 = (opts.input_width + 2*opts.padding - opts.kernel_size)/opts.stride + 1
    h_0 = (opts.input_height + 2*opts.padding - opts.kernel_size)/opts.stride + 1
    tflops = 2 * (w_0*h_0) * opts.kernel_size * opts.kernel_size * opts.channel_size * opts.filter_number * opts.batch_size * opts.batches_per_step * opts.loop_steps
    # print(f"iteration_report: tflops={tflops}")
    tflops_forw = tflops/time/10**12
    g_tflops_forw += tflops_forw
    tflops_forw_avg = g_tflops_forw / g_iterations

    itemsPerSec = opts.batch_size * opts.batches_per_step * opts.loop_steps / time
    g_items_per_second += itemsPerSec
    items_per_second_avg = g_items_per_second / g_iterations

    SecsPerIteration.append(time)
    ItemsPerSec.append(itemsPerSec)
    Tflops.append(tflops_forw)

    return "{:5f} items/sec, {:10f} TFLOPS; {:5f} items/sec average, {:10f} TFLOPS average".format(itemsPerSec, tflops_forw, items_per_second_avg, tflops_forw_avg)

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
          " Batch size {}.\n"
          " Batches per Step {}.\n"
          " Steps {}.\n"
          " Loop steps {}.\n"
          " Input width {}.\n"
          " Input height {}.\n"
          " Channel {}.\n"
          " Kernel num {}.\n"
          " Kernel size {}.\n"
          " Padding {}.\n"
          " Stride {}.\n"
          .format(
              {"infer": "Inference", "eval": "Evaluation",
                  "train": "Training"}[opts.mode],
              opts.batch_size,
              opts.batches_per_step,
              opts.steps,
              opts.loop_steps,
              opts.input_width,
              opts.input_height,
              opts.channel_size,
              opts.filter_number,
              opts.kernel_size,
              opts.padding,
              opts.stride))


    mode = {"infer": "Inference", "eval": "Evaluation", "train": "Training"}[opts.mode]
    print(f"\n\nPopART Convolutional layer {mode} Synthetic benchmark.\n")
    print("mfg,mode,dtype,sort,kernel,Batch_size,Batches_per_Step,Steps,Loop_Steps,Input_width,Input_height,Channel,Kernel_num,Kernel_size,Padding,Stride,Compile_Time,sec_per_iter,spare,items_per_second,mfg_TFLOPS,end_to_end")
    mfg = 'Graphcore'
    if opts.dtype == 'FLOAT16':
        sort = 'F16'
    else:
        sort = 'F32'

    print(f'{mfg},{mode},{opts.dtype},{sort},Kernel {opts.kernel},{opts.batch_size},{opts.batches_per_step},{opts.steps},{opts.loop_steps},{opts.input_width},{opts.input_height},{opts.channel_size},{opts.filter_number},{opts.kernel_size},{opts.padding},{opts.stride}')
    np.random.seed(42)
    run(module, opts)


    SecsPerIteration    = SecsPerIteration[opts.warmup:]
    ItemsPerSec         = ItemsPerSec[opts.warmup:]
    Tflops              = Tflops[opts.warmup:]
    print(f"{Average(SecsPerIteration)}    sec/itr.   {Average(ItemsPerSec)} items/sec,   {Average(Tflops)} TFLOPS")
