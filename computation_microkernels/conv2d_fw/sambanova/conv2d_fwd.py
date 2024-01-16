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
    parser.add_argument('--w',
                        type=int)
    parser.add_argument('--h',
                        type=int)
    parser.add_argument('--c',
                        type=int)
    parser.add_argument('--n',
                        type=int)
    parser.add_argument('--k',
                        type=int)
    parser.add_argument('--s',
                        type=int)
    parser.add_argument('--r',
                        type=int)
    parser.add_argument('--pad_w',
                        type=int)
    parser.add_argument('--pad_h',
                        type=int)
    parser.add_argument('--wstride',
                        type=int)
    parser.add_argument('--hstride',
                        type=int)
    parser.add_argument('--has-bias',
                        action='store_true',
                        help="Specify whether the linear nodes have bias. Default: No")
    parser.add_argument('--repeat',
                        type=int,
                        default=0,
                        help="Number of times to repeat this node in the network"
                        "Eg: 10")
    parser.add_argument('--activations',
                        nargs='+',
                        help="activations of linears, it can be a single str or a list of str. "
                        "Eg: --activations relu or --activations relu relu."
                        "If it is a single string, it will be broadcast to all linears")
    parser.add_argument('--repeat-dims',
                        nargs='+',
                        help="Specify dimensions for each repeated linear layer"\
                             "Eg: --repeat 3 --repeat-dims 1024 513 257"
                             "If --repeat > 1 and no --repeat-dims are specified, the out-features are repeated"
                             "'repeat' number of times")
    parser.add_argument('--repeat-pattern',
                        nargs='+',
                        help=f"Pattern of dimensions to repeat. This is helpful if you "
                        f"want a linear net with a repeating pattern "
                        f"Eg: --in-features 32 --out-features 64 --repeat 5 --repeat-pattern 1024 4096 will"
                        f"create a linear net with "
                        f" LinearNet((linears): ModuleList( "
                        "    (0): Linear(in_features=32, out_features=64, bias=False) "
                        "    (1): Linear(in_features=64, out_features=1024, bias=False) "
                        "    (2): Linear(in_features=1024, out_features=4096, bias=False) "
                        "    (3): Linear(in_features=4096, out_features=1024, bias=False) "
                        "    (4): Linear(in_features=1024, out_features=4096, bias=False) "
                        "    (5): Linear(in_features=4096, out_features=1024, bias=False) "
                        "    ) "
                        "  ) ")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="Momentum")
    parser.add_argument('--weight-decay', type=float, default=0.0001, help="Weight decay")
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('--dtype',
                        type=str,
                        default='float32',
                        choices=['float32', 'bfloat16'],
                        help='Data type of weights and inputs')
    parser.add_argument('--tensormem',
                        type=str,
                        default='host',
                        choices=['host', 'ddr'],
                        help=f"Compile time hint to specify tensor location during execution. "
                        f"Setting this to 'ddr' improves run-graph-only performance "
                        f"for this workload.  Otherwise, the default value 'host' is preferred.")
    parser.add_argument('--max-grad-norm-clip', type=float, default=None, help=f"Gradient clipping norm threshold.")
    parser.add_argument('--output-threshold', type=float, default=0.02, help=f"Absolute threshold of output tensor")
    parser.add_argument('--grad-threshold', type=float, default=0.05, help=f"Absolute threshold of grad tensors")


class ConvNet(nn.Module):
    def __init__(
            self,
            w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride
    ):
        super().__init__()

        self.layer = nn.Conv2d(in_channels=int(c),
                               out_channels=int(k),
                               kernel_size=(int(s), int(r)),
                               stride=(int(wstride), int(hstride)),
                               padding=(int(pad_w), int(pad_h)),
                               bias=False)
        self.output_mem = 'host'

    def forward(self, inputs):
        out1 = self.layer(inputs)
        out2 = self.layer(inputs)
        out3 = self.layer(inputs)
        out4 = self.layer(inputs)
        out5 = self.layer(inputs)
        if self.output_mem == 'host':
            out1.host_memory = True
        else:
            out1.host_memory = False
        return out1, out2, out3, out4, out5


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
                    tflops_forw = tflops/(sum(forward_pass_time)/n_iters/5)/(10**12) #tflops
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
