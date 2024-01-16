import os

import yaml
from modelzoo.common.tf.run_utils import is_cs

_curdir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_PATH = os.path.join(_curdir, "configs/params.yaml")

try:
    from cerebras.pb.common.tri_state_pb2 import TS_DISABLED
    from cerebras.pb.stack.full_pb2 import FullConfig
except ImportError:
    pass  # non-cbcore run

def get_params(params_file=DEFAULT_YAML_PATH, config="dense"):
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
    if config in params:
        params = params[config]
    else:
        raise AssertionError(
            f"Config {config} does not exist in params: {params}"
        )

    # input
    params["input"]["batch_size"] = params["input"].get("batch_size", 512)
    params["input"]["feature_shape"] = params["input"].get("feature_shape", [128,])
    params["input"]["num_classes"] = params["model"].get("num_classes", 16)
    # model
    params["model"]["hidden_size"] = params["model"].get("hidden_size", 2048)
    params["model"]["depth"] = params["model"].get("depth", 16)
    # runconfig
    params["runconfig"]["mode"] = params["runconfig"].get("mode", "train")
    params["runconfig"]["num_steps"] = params["runconfig"].get(
        "num_steps", 10000
    )
    params["fabric"] = params.get("fabric_json", None)

    params["optimizer"]["max_gradient_value"]= None
    params["optimizer"]["max_local_gradient_norm"]=None
    return params


def get_custom_stack_params(params):
    if params["model"]["model_name"] == "RNNModel":
        return get_custom_stack_params_rnn(params)
    else:
        return get_custom_stack_params_dense(params)


def get_custom_stack_params_dense(params):
    stack_params = {}

    from cerebras.pb.stack.full_pb2 import FullConfig

    config = FullConfig()
    config.matching.optimize_graph.dedicated_buffers.enable_fc = True
    config.matching.kernel.dp_kernel_list.extend(["FcLayer"])

    if params["runconfig"]["multireplica"]:
        config.target_num_replicas = -1
        os.environ["CEREBRAS_CUSTOM_MONITORED_SESSION"] = "True"

    stack_params["config"] = config

    return stack_params

def get_custom_stack_params_rnn(params):
    stack_params = dict()
    runconfig_params = params["runconfig"]
    use_cs = is_cs(runconfig_params)
    if (
        use_cs
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        stack_params["config"] = set_custom_config_rnn(FullConfig(), params)
        stack_params["ir_mode"] = "xla"
        return stack_params


def set_custom_config_rnn(config, params):
    model_params = params["model"]
    if (
        model_params.get("residual_type",None) == "long"
        and model_params.get("use_residuals",False)
    ) or model_params.get("share_embedding",False):
        config.matching.optimize_graph.ubatch_size = 0
    if (
        model_params.get("enable_layer_norm_before_dropout", False) or 
        model_params.get("enable_layer_norm_after_dropout", False)
    ):
        config.placement.match_port_format.prefer_dense_packets = TS_DISABLED
    #config.matching.kernel.enable_oned_conversion = True
    config.matching.kernel.dp_kernel_list.extend(["FcLayer"])
    return config
