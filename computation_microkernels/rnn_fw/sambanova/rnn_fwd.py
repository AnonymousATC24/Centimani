import argparse
import random
import os
import sys
import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import Optional, Tuple, List, Any, Union

import sambaflow.samba.utils as sn_utils

from sambaflow import samba
from sambaflow.samba.sambatensor import SambaTensor
from sambaflow.samba.lazy_param import lazy_param
from sambaflow.samba.materialize import materialize
from sambaflow.samba.utils.argparser import parse_app_args

import time

def get_inputs(args) -> Tuple[samba.SambaTensor, ...]:
    dim_input = (args.n, args.c, args.w, args.h)
    if args.dtype == 'bfloat16':
        input_tensor = SambaTensor(torch.rand(dim_input).bfloat16(), name='input_tensor', batch_dim=0)
    else:
        input_tensor = SambaTensor(torch.rand(dim_input), name='input_tensor', batch_dim=0)

    if not args.inference:
        input_tensor.requires_grad_(True)

    return input_tensor,


def get_optim(params, args) -> Any:

    if args.inference:
        return None

    else:
        optim = None

        # Default Optimizer: SGD
        if args.optim == 'sgd':
            optim = samba.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Override with Adam if specified
        elif args.optim == 'adam':
            optim = samba.optim.AdamW(params,
                                      lr=args.lr,
                                      betas=(0.997, 0.997),
                                      weight_decay=args.weight_decay,
                                      max_grad_norm=args.max_grad_norm_clip)

        return optim


def add_args(parser: argparse.Namespace):
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Set batch size.')
    parser.add_argument('--time-step', type=int, default=1,
                        help='Set number of recurrent steps.')
    parser.add_argument('--input-size', type=int, default=32,
                        help='Set number of units in input layer.')
    parser.add_argument('--hidden-size', type=int, default=32,
                        help='Set number of units in hidden layer.')
    parser.add_argument('--number-layers', type=int, default=1,
                        help='Set number of layers.')
    #parser.set_defaults(batches_per_step=1000, steps=1,
    #                    mode='infer', auto_sharding=True)



    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('--dtype',
                        type=str,
                        default='float32',
                        choices=['float32', 'bfloat16'],
                        help='Data type of weights and inputs')



class LstmModel(nn.Module):
    """
        Simple LSTM model
    """

    def __init__(self, num_layers, input_size, hidden_size):
        """
        constructor just calls super.
        """
        super(LstmModel, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )

    def forward(self, input, h0, c0):
        result = self.rnn(input, (h0, c0))
        return result


class ConvNet(nn.Module):
    def __init__(
            self,
            w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride
    ):
        super().__init__()

        # input is the sequence which is fed into the network. It should be of size (seq_len, batch, 
        # input_size). If batch_first=True, the input size is (batch, seq_len, input_size).

        # h_0 is the initial hidden state of the network. It is of the size 
        # (num_layers * num_directions, batch, input_size) where num_layers is the number of 
        # stacked RNNs. num_directions = 2 for bidirectional RNNs and 1 otherwise.        

        self.layer = nn.Conv2d(in_channels=int(c),
                               out_channels=int(k),
                               kernel_size=(int(s), int(r)),
                               stride=(int(wstride), int(hstride)),
                               padding=(int(pad_w), int(pad_h)),
                               bias=False)
        self.output_mem = 'host'

    def forward(self, inputs):
        out = self.layer(inputs)
        if self.output_mem == 'host':
            out.host_memory = True
        else:
            out.host_memory = False
        return out


def main(argv: List[str]):
    # Set random seed for reproducibility.
    sn_utils.set_seed(0)

    # Get common args and any user added args.
    args = parse_app_args(argv=argv, common_parser_fn=add_args, dev_mode=True)

    # Get the inputs
    inputs = get_inputs(args)

    with lazy_param():
        # Instantiate the model.
        model = ConvNet(args.w, args.h, args.c, args.n, args.k, args.s, args.r, args.pad_w, args.pad_h, args.wstride, args.hstride)
        if args.dtype == 'bfloat16':
            model.bfloat16()

    samba.from_torch_(model)

    # Get the optimizer
    optim = get_optim(model.parameters(), args)

    if args.command == 'compile':
        samba.session.compile(model,
                              inputs,
                              optim,
                              name='conv2d_net',
                              init_output_grads=not args.inference,
                              app_dir=sn_utils.get_file_dir(__file__),
                              config_dict=vars(args))

        if samba.is_internal() and args.resource_report_path:
            compile_dir = os.path.join(args.output_folder, args.pef_name)
            sn_utils.create_mac_resource_report(
                compile_report_file_path=str(args.verbose_report),
                mac_resource_file_path=str(os.path.join(compile_dir, "mac_gen", "mac_resources.json")),
                prism_node_reports_dir=str(os.path.join(compile_dir, "analytical_models")),
                output_file_path=str(args.resource_report_path))
            assert os.path.isfile(
                args.resource_report_path), f'Failed to generate MAC resource report at {args.resource_report_path}'
    else:
        if args.command in ['test', 'measure-performance', 'run', 'measure-sections']:
            if args.num_spatial_batches > 1:
                args.batch_size *= args.num_spatial_batches
                inputs = SambaTensor(inputs[0].torch().repeat(args.num_spatial_batches, 1), name='input', batch_dim=0),
                if args.tensormem == 'host':
                    inputs.host_memory = True
                else:
                    inputs.host_memory = False

            if args.command == 'test':
                # materialize the model under test to run model on CPU
                with materialize(model) as model:
                    # put trace graph in materialize context to make sure the model is initialized once for numeric
                    # reference from torch.
                    sn_utils.trace_graph(model,
                                         inputs,
                                         optim=optim,
                                         init_output_grads=not args.inference,
                                         pef=args.pef,
                                         mapping=args.mapping)
                    # Test numerical correct between CPU and RDU, need to materialize the model for torch CPU
                    test(args, model, inputs)

            else:
                sn_utils.trace_graph(model,
                                     inputs,
                                     optim=optim,
                                     init_output_grads=not args.inference,
                                     pef=args.pef,
                                     mapping=args.mapping)
                if args.command == 'measure-performance':
                    # Get inference latency and throughput statistics
                    sn_utils.measure_performance(model,
                                                 inputs,
                                                 args.batch_size,
                                                 args.n_chips,
                                                 args.inference,
                                                 run_graph_only=args.run_graph_only,
                                                 n_iterations=args.num_iterations,
                                                 json=args.json,
                                                 compiled_stats_json=args.compiled_stats_json,
                                                 data_parallel=args.data_parallel,
                                                 reduce_on_rdu=args.reduce_on_rdu,
                                                 use_sambaloader=True,
                                                 min_duration=args.min_duration)

                elif args.command == 'run':
                    samba.session.run(inputs, section_types=['fwd'])
                    #samba.session.run(inputs, section_types=['bckwd'])
                    n_iters = 100
                    forward_pass_time = []
                    print("run starts")
                    start_time_forward = time.time()
                    for loop in range(n_iters):
                        samba.session.run(inputs, section_types=['fwd'])
                        #samba.session.run(inputs, section_types=['bckwd'])
                        #samba.session.run(inputs, section_types=['fwd', 'bckwd'])
                    end_time_forward = time.time()
                    forward_pass_time.append(end_time_forward - start_time_forward)
                    print("run ends")

                    w_0 = (args.w + 2*args.pad_w - args.s)/args.wstride + 1
                    h_0 = (args.h + 2*args.pad_h - args.r)/args.hstride + 1
                    tflops = 2 * (w_0*h_0) * args.s * args.r * args.c * args.k * args.n
                    tflops_forw = tflops/(sum(forward_pass_time)/n_iters)/(10**12) #tflops
                    print(tflops)
                    print(sum(forward_pass_time))
                    print("tflops: %f"%tflops_forw)
                    print("SN,Training,%s,Conv2d_fwd,%d,100,1,%d,%d,%d,%d,%d,%d,%d,0.0,%f,None,%f,%f,%f" % ("dtype", args.n, args.w, args.h, args.c, args.k, args.s, args.pad_w, args.wstride, (sum(forward_pass_time)/n_iters)/args.n, args.n/(sum(forward_pass_time)/n_iters), tflops_forw, (sum(forward_pass_time)/n_iters)/args.n))


                elif args.command == 'measure-sections':
                    sn_utils.measure_sections(model,
                                              inputs,
                                              num_sections=args.num_sections,
                                              n_iterations=args.num_iterations,
                                              batch_size=args.batch_size,
                                              data_parallel=args.data_parallel,
                                              reduce_on_rdu=args.reduce_on_rdu,
                                              json=args.json,
                                              min_duration=args.min_duration)


if __name__ == '__main__':
    main(sys.argv[1:])
