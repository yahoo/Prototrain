# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

import tensorflow as tf
from pprint import pprint
from config import config
from commons.model_utils import ModelDict
from commons.data_utils import json_line_dataset


def build(inputs=None, train_phase=tf.constant(False)):

    # if no inputs provided via training code, assume we're in
    # inference mode and create placeholders
    if inputs is None:
        inputs = {
            "examples": tf.placeholder(tf.float32, [None, 2])
        }

    # create model
    model = ModelDict()
    model.inputs = inputs
    model.examples = inputs["examples"]

    reuse_weights = tf.get_variable_scope().reuse

    model.fc0 = tf.layers.dense(model.examples,
                                units=10,
                                activation=tf.nn.relu,
                                use_bias=False,
                                reuse=reuse_weights)
    model.fc1 = tf.layers.dense(model.last,
                                units=50,
                                activation=tf.nn.relu,
                                use_bias=False,
                                reuse=reuse_weights)
    model.bn1 = tf.layers.batch_normalization(model.last, training=train_phase)
    model.fc2 = tf.layers.dense(model.last,
                                units=50,
                                activation=tf.nn.relu,
                                use_bias=False,
                                reuse=reuse_weights)
    model.fc3 = tf.layers.dense(model.last,
                                units=10,
                                use_bias=True,
                                reuse=reuse_weights)
    model.probs = tf.nn.softmax(model.last)

    # add loss if labels available
    if "labels" in inputs:
        model.labels = inputs["labels"]
        model.labels_ohe = tf.one_hot(model.last, depth=10)
        model.loss_example = tf.nn.softmax_cross_entropy_with_logits(
                                                             labels=model.labels_ohe,
                                                             logits=model.fc3,
                                                             dim=-1,
                                                             name=None)

        model.loss = tf.reduce_mean(model.last)
    return model


def datasets():
    trn_path   = config['dataset.trn']
    vld_path   = config['dataset.vld']
    tst_path   = config['dataset.tst']
    batch_size = config['model.batch_size']

    def parse(record):
        # example preprocessing, cast var from int to float
        examples = tf.cast(record["example"], tf.float32) / 6
        labels = record['label']
        return {
            "examples": examples,
            "labels": labels
        }

    def load_dataset(path, shuffle=True, repeat=True):
        ds = json_line_dataset(path,
                               output_types = {
                                   "example": tf.int64,
                                   "label": tf.int64
                               },
                               output_shapes = {
                                   "example": tf.TensorShape([2]),
                                   "label": tf.TensorShape(None)
                               },
                               repeat=repeat)
        ds = ds.map(parse, num_parallel_calls=64)
        ds = ds.batch(batch_size)
        if shuffle:
            ds = ds.shuffle(buffer_size=100)
        ds = ds.prefetch(10)
        return ds

    trn_ds = load_dataset(trn_path)
    vld_ds = load_dataset(vld_path)
    tst_ds = load_dataset(tst_path, shuffle=False, repeat=False)

    return {"trn": trn_ds,
            "vld": vld_ds,
            "tst": tst_ds}
