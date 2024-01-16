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

"""
UNet model to be used with TF Estimator
"""
import tensorflow as tf
from modelzoo.common.tf.layers.RNNEncoderBlock import RNNEncoderBlock
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.CrossEntropyFromLogitsLayer import (
    CrossEntropyFromLogitsLayer,
)
from modelzoo.common.tf.layers.PoolerLayer import PoolerLayer
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from tensorflow.python.keras.layers import Flatten, concatenate
from tensorflow.compat.v1.losses import Reduction
from modelzoo.common.tf.layers.LSTMCell import LSTMCell
from modelzoo.common.tf.layers.RNNLayer import RNNLayer

class RNNModel(TFBaseModel):
    """
    RNN model with CrossEntropy loss
    """

    def __init__(self, params):
        super(RNNModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        self.num_classes = params["input"]["num_classes"]


        ### Model params
        mparams = params["model"]

        
        # CS util params for layers
        self.boundary_casting = mparams["boundary_casting"]
        self.tf_summary = mparams["tf_summary"]

        self.mixed_precision = mparams["mixed_precision"]

        # Model trainer
        self.trainer = Trainer(
            params=params["optimizer"],
            tf_summary=self.tf_summary,
            mixed_precision=self.mixed_precision,
        )
        # RNN block, by default contains RNN + dropout.
        
        self.rnn_encoder_block = RNNEncoderBlock(
            hidden_size=mparams['hidden_size'],
            encoder_depth=mparams['depth'],
            dropout_rate=mparams.get('dropout_rate',0.3),
            rnn_cell=mparams['rnn_cell'],
            rnn_use_bias=mparams['rnn_use_bias'],
            rnn_activation=mparams['rnn_activation'],
            rnn_recurrent_activation=mparams['rnn_recurrent_activation'],
            rnn_unit_forget_bias=mparams['rnn_unit_forget_bias'],
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.policy,
        )
       

        self.max_seq_len = params["input"]["feature_shape"][0]
        self.hidden_size = mparams['hidden_size']


    def build_model(self, features, mode):
        # Get input image.
        y = features['y']
        
        y = DenseLayer(
            units=self.hidden_size,
            use_bias=True,
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.policy,
        )(y)
        y_lens = features['y_lens']
        mask = tf.sequence_mask(
            y_lens, self.max_seq_len, dtype=tf.bool,
        )
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        output = self.rnn_encoder_block(
            y, mask=mask, is_training=is_training
        )

        ##### Output
        
        logits = DenseLayer(
            units=self.num_classes,
            use_bias=True,
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.policy,
        )(output)
        
        return logits

    def build_total_loss(self, logits, features, labels, mode):

        softmax_ce_layer = CrossEntropyFromLogitsLayer(
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.policy,
            name='softmax_ce_loss',
        )
        
        cross_entropy = softmax_ce_layer(labels, logits=logits)
        y_lens = features['y_lens']
        cross_entropy_mask = tf.sequence_mask(
            y_lens, self.max_seq_len, dtype=cross_entropy.dtype,
        )
        cross_entropy = cross_entropy * cross_entropy_mask
        
        # Average loss per sequence
        cross_entropy_loss = tf.reduce_sum(
            tf.cast(cross_entropy, tf.float32)
        ) / tf.cast(
            features['y'].shape[0], tf.float32  # batch_size
        )
        cross_entropy_loss = tf.cast(cross_entropy_loss, cross_entropy.dtype)
        self._write_summaries(
            features, cross_entropy_loss,
        )

        return cross_entropy_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.trainer.build_train_ops(total_loss)
    
    def build_eval_metric_inputs(self, model_outputs, labels, features):
        """
        Build inputs for eval metrics computations.
        :param model_outputs: Model output tensor returned by call method.
        :param labels: (2D tensor) decoder target token sequence.
        :param features: Dictionary of input features.

        :return: `eval_metric_inputs`: tuple containing:
                -- `predictions`: predicted labels for each token;
        """
        predictions = tf.argmax(model_outputs, axis=-3, output_type=tf.int32)

        return (predictions,)

    def build_eval_metric_ops(self, eval_metric_inputs, labels, features):
        """
        Compute Transformer eval metrics - BLEU score.
        :param `eval_metric_inputs`: tuple containing:
            -- `predictions`: predicted labels for each token;
            Tensor of shape (batch_size, tgt_max_sequence_length).
        :param labels: Tensor of shape (batch_size, tgt_max_sequence_length).
                Contains expected reference translation labels.
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        return {}
    def _write_summaries(self, features, total_loss):
        """
        Write train metrics summaries
        """

        total_loss = tf.cast(total_loss, tf.float32)
        # Log losses: total, nsp, and mlm
        tf.compat.v1.summary.scalar('train/cost', total_loss)
