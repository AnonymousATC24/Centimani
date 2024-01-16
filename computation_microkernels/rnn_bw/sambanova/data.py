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
Synthetic data generation for simple benchmarks
"""


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _generator(params):
    """
    Generator to synthetically create features and labels.
       Feature shape: params[train_input][feature_shape]
       Labels shape: [1]
    """
    for _i in range(params["input"]["batch_size"]):
        input_data = np.random.uniform(size=params["input"]["feature_shape"])
        labels = np.random.randint(
            low=1, 
            high=params["input"]["num_classes"],
            size=params["input"].get("label_shape",[]),
        )
        if params['model']['model_name']=='RNNModel':
            features = {}
            features['y']=input_data
            features['y_lens']=params["input"]["feature_shape"][0]
        else:
            features = input_data
        yield features, labels


def dataset_from_slices(params):
    """
    Creates dataset using Dataset.from_tensor_slices
       Feature shape: params[train_input][feature_shape]
       Labels shape: [1]
    """
    dtype = tf.float16 if params["model"]["mixed_precision"] else tf.float32

    input_data = tf.random.stateless_uniform(
        shape=[params["input"]["batch_size"],] + params["input"]["feature_shape"],
        seed=[0, 10],
        dtype=dtype,
    )
    if params['model']['model_name']=='RNNModel':
        features = {}
        features['y']=input_data
        features['y_lens']=[params["input"]["feature_shape"][0],]*params["input"]["batch_size"] 
    else:
        features = input_data
    

    labels = tf.random.stateless_uniform(
        shape=[params["input"]["batch_size"],] + params["input"].get("label_shape",[]),
        maxval=params["input"]["num_classes"],
        seed=[0, 10],
        dtype=tf.int32,
    )

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset


def dataset_from_generator(params):
    """
    Creates dataset using Dataset.from_generator
       Feature shape: params[train_input][feature_shape]
       Labels shape: [1]
    """
    dtype = tf.float16 if params["model"]['mixed_precision'] else tf.float32

    dataset = tf.data.Dataset.from_generator(
        lambda: _generator(params=params),
        output_types=(dtype, tf.int32),
        output_shapes=(
            params["input"]["feature_shape"],
            params["input"].get("label_shape",[]),
        ),
    )
    return dataset


def input_fn(params, mode=tf.estimator.ModeKeys.TRAIN):
    """
    Input function that returns a synthetic tf.data.Dataset
    choosing from generator or tensor slices.
    """

    dataset = (
        dataset_from_generator(params)
        if params["input"].get("use_generator", False)
        else dataset_from_slices(params)
    )
    dataset = dataset.repeat().batch(
        params["input"]["batch_size"], drop_remainder=True
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset



