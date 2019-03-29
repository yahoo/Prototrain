# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

import tensorflow as tf

def restore_available(sess, fpath,
                      verbose=1,
                      restore_global_step=False):
    """
    Restores all available variables within a checkpoint file.
    Ignores all variables that are in checkpoint but not in the graph.
    Ignores all variables that are in graph but not in checkpoint.
    Ingores all variables with mismatched shapes between graph and checkpoint.

    Args:
        sess: tf.Session
            Session to restore the variables to
        fpath: string
            Path to your checkpoint file
        restore_global_step: boolean
            If False, avoid restoring the "global_step" variable

    """
    with sess.graph.as_default():
        # check what variables are available in file
        reader = tf.train.NewCheckpointReader(fpath)
        ckpt_var_shape_map = reader.get_variable_to_shape_map()
        all_var_names_in_ckpt = ckpt_var_shape_map.keys()

        # check all variables available in graph
        all_vars_in_graph = dict((v.op.name, v) for v in tf.global_variables())
        all_var_names_in_graph = all_vars_in_graph.keys()

        # variables in either graph and ckpt file
        all_vars_names = set(all_var_names_in_graph).union(all_var_names_in_ckpt)

        # construct dict with only variables that are available in both
        names_to_vars = {}
        not_in_graph = []
        not_in_ckpt = []
        not_same_shape = []
        excluded = []
        loaded = []


        for varname in sorted(all_vars_names):
            if 'global_step' == varname and not restore_global_step:
                excluded.append(varname)
                continue
            if varname not in all_var_names_in_graph:
                not_in_graph.append(varname)
                continue
            if varname not in all_var_names_in_ckpt:
                not_in_ckpt.append(varname)
                continue
            if tuple(ckpt_var_shape_map[varname]) != tuple(all_vars_in_graph[varname].shape):
                not_same_shape.append(varname)
                continue
            loaded.append(varname)
            names_to_vars[varname] = all_vars_in_graph[varname]

    # print what was loaded, excluded, or not found
    if verbose > 0:
        for varname in loaded:
            print "INFO: Loaded %s" % varname
        for varname in excluded:
            print "INFO: Excluded  %s" % varname
        for varname in not_in_ckpt:
            print "WARN: Not in ckpt: %s" % varname
        for varname in not_in_graph:
            print "WARN: Not in graph: %s" % varname
        for varname in not_same_shape:
            print "WARN: Not same shape: %s checkpoint=%s vs graph=%s" \
                % (varname, ckpt_var_shape_map[varname], all_vars_in_graph[varname].shape)

    restorer = tf.train.Saver(dict(names_to_vars))
    restorer.restore(sess, fpath)


def read_variables(fpath):
    """
    loads variables from checkpoint

    Args:
        path to checkpoint ".ckpt"
    Returns:
        dictionary where key is the variable name and value is the variable
        value

    """
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(fpath)
    var_to_shape_map = reader.get_variable_to_shape_map()
    variables = {}
    for k in var_to_shape_map.keys():
        variables[k] = reader.get_tensor(k)
    return variables
