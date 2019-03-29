# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

import os
import re
import tensorflow as tf
from pprint import pformat


def setup_experiment_dir(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)


def save_configuration(config, experiment_dir):
    import pickle as pkl
    config_file_machine_readable_path = os.path.join(experiment_dir,
                                        "config_machine_readable.pkl")
    config_file_human_readable_path = os.path.join(experiment_dir,
                                        "config_human_readable.txt")
    with open(config_file_machine_readable_path, "wb") as fh:
        pkl.dump(config, fh)
    with open(config_file_human_readable_path, "w") as fh:
        fh.write(pformat(config))




def setup_summary_writers(experiment_dir):
    G = tf.get_default_graph()
    summary_dir = os.path.join(experiment_dir, "summaries")

    trn_summary_dir = os.path.join(summary_dir, "trn")
    vld_summary_dir = os.path.join(summary_dir, "vld")

    if not os.path.exists(trn_summary_dir):
        os.makedirs(trn_summary_dir)
    if not os.path.exists(vld_summary_dir):
        os.makedirs(vld_summary_dir)

    writers = {}
    writers['trn'] = tf.summary.FileWriter(trn_summary_dir, graph=G)
    writers['vld'] = tf.summary.FileWriter(vld_summary_dir, graph=G)
    return writers





def clip_gradients(grads, minval=None, maxval=None):
    minval = minval if minval is not None else -1.0
    maxval = maxval if maxval is not None else 1.0
    grads_clipped = [(tf.clip_by_value(grad, minval, maxval), var)
                     for grad, var in grads]
    return grads_clipped

def average_gradients(grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def setup_learning_rate(config, global_step):
    """
    Setup learning rate schedule based on configuration flags

    Args:
        config: config object
        global_step: tf.int64

    Returns:
        learning_rate: tensor
    """

    # setup learning rate schedule
    lr_schedule = config['model.lr_schedule.method']
    if lr_schedule == 'step':
        boundaries = config['model.lr_schedule.step.boundaries']
        values = config['model.lr_schedule.step.values']
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    boundaries=boundaries,
                                                    values=values,
                                                    name="learning_rate")

    elif lr_schedule == 'exp':
        decay_steps = config['model.lr_schedule.exp.decay_steps']
        decay_rate = config['model.lr_schedule.exp.decay_rate']
        learning_rate_value = config['model.lr_schedule.exp.value']
        learning_rate = tf.train.exponential_decay(learning_rate_value,
                                                   global_step=global_step,
                                                   decay_steps=decay_steps,
                                                   decay_rate=decay_rate,
                                                   name="learning_rate")

    elif lr_schedule == 'constant':
        learning_rate_value = config['model.lr_schedule.constant.value']
        learning_rate = tf.Variable(learning_rate_value,
                                    trainable=False,
                                    name="learning_rate")
    else:
        raise ValueError("Unrecognized learning_rate_schedule")

    return learning_rate

def setup_optimizer(config, learning_rate):
    """
    Setup optimizer based on configuration flags

    Args:
        config: config object
        learning_rate: tf.float32

    Returns:
        optimizer:
            Optimizer object
    """

    # setup optimizer
    optimizer_method = config['model.optimizer.method']
    if optimizer_method == 'msgd':
        momentum = config['model.optimizer.msgd.momentum']
        use_nesterov = config['model.optimizer.msgd.use_nesterov']
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=momentum,
                                               use_nesterov=use_nesterov)
    elif optimizer_method == 'adam':
        beta1 = config['model.optimizer.adam.beta1']
        beta2 = config['model.optimizer.adam.beta2']
        epsilon = config['model.optimizer.adam.epsilon']
        optimizer = tf.train.AdamOptimizer(learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
    elif optimizer_method == 'rmsprop':
        decay = config['model.optimizer.rmsprop.decay']
        momentum = config['model.optimizer.rmsprop.momentum']
        epsilon = config['model.optimizer.rmsprop.epsilon']
        centered = config['model.optimizer.rmsprop.centered']
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              decay=decay,
                                              momentum=momentum,
                                              epsilon=epsilon,
                                              centered=centered)
    elif optimizer_method == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError("Unrecognized solver method")
    return optimizer


def filter_trainable_variables(config):
    if config['model.trainable_vars.filter_fn']:
        filter_fn = config['model.trainable_vars.filter_fn']
    elif config['model.trainable_vars.filter_regex']:
        filter_regex = re.compile(config['model.trainable_vars.filter_regex'])
        filter_fn = lambda vname: filter_regex.match(vname) is not None
    else:
        filter_fn = lambda vname: True

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    vars = [v for v in vars if filter_fn(v.name)]
    return vars

def filter_weight_decay_variables(config):
    filter_fn = config['model.weight_decay.filter_fn']
    filter_regex = config['model.weight_decay.filter_regex']
    if filter_fn:
        pass
    elif filter_regex:
        filter_regex = re.compile(filter_regex)
        filter_fn = lambda vname: filter_regex.match(vname) is not None
    else:
        filter_fn = lambda vname: True

    vars = tf.global_variables()
    vars = [v for v in vars if filter_fn(v.name)]
    return vars

def get_dataset_handle(ds, sess):
    return sess.run(ds.make_one_shot_iterator().string_handle())

def discard_empty_gradients(grads_and_vars):
    return [(g,v) for g,v in grads_and_vars if g is not None]

def setup_global_step(config):
    return tf.Variable(config['model.starting_step'],
                       name='global_step',
                       trainable=False,
                       dtype=tf.int64)

def split_batch(inputs, n):
    splits = []
    for i in range(n):
        splits.append({})

    for key, tensor in inputs.items():
        split = tf.split(axis=0,
                         num_or_size_splits=n,
                         value=tensor,
                         name="%s_split" % key)
        for i in range(n):
            splits[i][key] = split[i]
    return splits
