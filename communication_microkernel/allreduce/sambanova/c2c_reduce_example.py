"""Build, compile and run C2C reduce collective op for fully connected A1.1 4-way topology."""

import argparse
import numpy as np

import groq.api as g
from groq.common import print_utils
from groq.common.config import config

import time

data_size = 6553600

def create_data(input_tensors, result_tensor):
    # Create random data for each input tensor.
    input_data = []
    for input_t in input_tensors:
        shape = input_t.shape
        nptype = input_t.dtype.to_nptype()
        data = (np.random.rand(*shape) * 10.0).astype(nptype)
        input_data.append(data)

    # Build input dictionary.
    inputs = {input_t.name: data for input_t, data in zip(input_tensors, input_data)}

    # Build oracle from src data.
    oracle_data = np.concatenate(np.array(input_data), axis=0).sum(axis=0)
    oracles = {result_tensor.name: oracle_data}

    return inputs, oracles


def check_results(results, oracles) -> bool:
    all_okay = True
    for result_key, result in results.items():
        oracle = oracles[result_key]
        try:
            np.testing.assert_allclose(result, oracle, rtol=1e-5)
        except AssertionError as exc:
            print_utils.err(f"Result mismatch for tensor '{result_key}' =>")
            print_utils.warn(f"Comparing (result, oracle): {exc}")
            all_okay = False
            continue
        print_utils.success(f"Result matched the oracle for tensor '{result_key}'")
    return all_okay


def create_input_tensors(shape, dtype, num_devs):
    # Create an input tensor for each device.
    input_tensors = []
    for dev_num in range(num_devs):
        with g.device(dev_num):
            mt = g.input_tensor(shape, dtype, name=f"c2c_inp_d{dev_num}")
            input_tensors.append(mt)
    return input_tensors


def build_program(user_config, pgm_pkg, prog_name):
    # Setup multi-chip topology and create a new program context.
    topo = g.configure_topology(config=user_config)
    print_utils.infoc(
        f"Building C2C program '{prog_name}' with '{topo.name}' topology ..."
    )
    pg_ctx = pgm_pkg.create_program_context(prog_name, topo)

    with pg_ctx:
        shape = (1, data_size)
        dtype = g.float32
        num_devs = pg_ctx.num_devices()

        # Initialize tensors.
        input_tensors = create_input_tensors(shape, dtype, num_devs)

        # Perform c2c reduce sum operation (defaults to sending result to device 0).
        result_tensor = g.c2c_reduce(input_tensors, op=g.C2CReduceOp.SUM, time=1000)
        result_tensor.set_program_output()

    return input_tensors, result_tensor


def get_config_from_topo_str(topo_str):
    if topo_str == "A11_2C":
        return g.TopologyConfig.FC2_A11_2_CHIP
    elif topo_str == "A11_4C":
        return g.TopologyConfig.FC2_A11_4_CHIP
    elif topo_str == "A14_2C":
        return g.TopologyConfig.DF_A14_2_CHIP
    elif topo_str == "A14_4C":
        return g.TopologyConfig.DF_A14_4_CHIP
    elif topo_str == "A14_8C":
        return g.TopologyConfig.DF_A14_8_CHIP
    else:
        # fallback to A11 4card topology
        return g.TopologyConfig.FC2_A11_4_CHIP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C2C collectives example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-g",
        "--group",
        type=str,
        default="A",
        help="Topology group TSPs belong to run this C2C program",
    )
    parser.add_argument(
        "--bringup", action="store_true", help="Bringup C2C links for given topology",
    )
    parser.add_argument(
        "--topo_str",
        type=str,
        default="A11_2C",
        choices=["A11_2C", "A11_4C", "A14_2C", "A14_4C", "A14_8C"],
        help="Topology type to run this C2C program: \n",
    )
    args = parser.parse_args()

    # Step 1: Instantiate a program package to store multi-chip (C2C)
    # or single-chip programs.
    pkg_name = "c2c_pkg"
    pkg_dir = config.get_tmp_dir(pkg_name)
    print_utils.infoc(f"Creating a program package '{pkg_name}' at '{pkg_dir}' ...")
    pgm_pkg = g.ProgramPackage(name=pkg_name, output_dir=pkg_dir)

    # Step 2: Build your multi-chip C2C program.
    prog_name = "c2c_reduce"
    user_config = get_config_from_topo_str(args.topo_str)
    input_tensors, result_tensor = build_program(user_config, pgm_pkg, prog_name)

    # You are free to add more multi-chip or single-chip programs to the package.

    # Step 3: Assemble all programs in the multi-device package.
    print_utils.infoc(f"Assembling multi-device package '{pkg_name}' ...")
    pgm_pkg.assemble()

    # Step 4 [Optional]: Bringup the c2c links before we run the program.
    # link up procedure is now embedded in multi_tsp_runner, it will
    # check for the link status before it begins invoking the programs
    # the user can force the linkup by passing --bringup option
    try:
        if args.bringup:
            print_utils.infoc("Bringup C2C topology ...")
            g.bringup_topology(group=args.group, topo_type=user_config.value)
    except Exception as e:  # pylint: disable=broad-except
        print_utils.infoc("Aborting, " + str(e))
        exit(1)

    # Step 5: Create a multi-tsp runner
    # Make sure to pass the program name to be executed.
    print_utils.infoc("Creating multi-tsp runner ...")
    try:
        runner = g.create_multi_tsp_runner(
            pkg_name,
            pkg_dir,
            prog_name,
            group=args.group,
            topo_type=user_config.value,
        )
    except Exception as e:  # pylint: disable=broad-except
        print_utils.infoc("Aborting, " + str(e))
        exit(1)

    # Step 6: Pass inputs to the runner and execute the program on HW.
    inputs, oracles = create_data(input_tensors, result_tensor)
    print_utils.infoc(f"Executing C2C program '{prog_name}' ...")
    try:
        start_time = time.time()
        for loop in range(100): 
            results = runner(**inputs)
        end_time = time.time()
        exec_time = end_time - start_time
        time_taken = exec_time / 100.0
        print("Execution time:%f" % exec_time)
        amount_bytes = data_size * 4
        bw = amount_bytes / time_taken / 1000000000;
        print("BW:%f GB/s" % bw)

    except Exception as e:  # pylint: disable=broad-except
        print_utils.infoc("Aborting, " + str(e))
        exit(1)

    # Validation: Compare against oracle.
    print_utils.infoc("Validating results ...")
    assert check_results(results, oracles), "Validation failed!"
