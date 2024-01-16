# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import sys

from anl_shared.template.tf.DenseModel import DenseModel
from anl_shared.template.tf.CNNModel import CNNModel
from anl_shared.template.tf.RNNModel import RNNModel
from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec


def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API
    """

    model = getattr(sys.modules[__name__], params["model"]["model_name"])(
        params
    )
    outputs = model(features,mode)

    loss = model.build_total_loss(outputs,features,labels,mode) 
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = model.build_train_ops(loss)
    elif not  mode == tf.estimator.ModeKeys.EVAL:
        raise ValueError(f"Mode {mode} not supported.")

    espec = CSEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
    )
    return espec
