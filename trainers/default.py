# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

import os
import sys
import json
import time
import tensorflow as tf
import numpy as np
import shutil
from argparse import ArgumentParser
from commons.checkpoint_utils import restore_available
from commons.train_utils import (
    setup_learning_rate,
    setup_optimizer,
    split_batch,
    discard_empty_gradients,
    filter_trainable_variables,
    filter_weight_decay_variables,
    average_gradients,
    clip_gradients,
    get_dataset_handle,
    setup_experiment_dir,
    setup_summary_writers,
    save_configuration
)


def train(model, args):

    # ===================
    # Get config settings
    # ===================
    config = model.config
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    num_gpus = 1 if num_gpus == 0 else num_gpus
    num_steps = config['model.num_steps']
    exp_base_dir = config['trainer.experiment_base_dir']
    exp_name = config['trainer.experiment_name']
    exp_dir = os.path.join(exp_base_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoint")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    weights_load_path = config['model.weights_load_path']
    validation_step_interval = config['trainer.validation_step_interval']
    summary_save_interval = config['trainer.summary_save_interval']
    print_interval = config['trainer.print_interval']
    checkpoint_save_interval = config['trainer.checkpoint_save_interval']
    max_checkpoints_to_keep = config['trainer.max_checkpoints_to_keep']
    weight_decay_value = config['model.weight_decay.value']
    clip_gradients_enabled = config['model.clip_gradients']
    clip_gradients_minval = config['model.clip_gradients.minval']
    clip_gradients_maxval = config['model.clip_gradients.maxval']

    batch_size = config['model.batch_size']
    starting_step = 0

    # ensure batch_size is divisible by number of GPUS
    assert (batch_size % num_gpus) == 0, \
        "batch_size must be divisible by num_gpus"

    if args.fresh and os.path.exists(exp_dir):
        question = "Are you sure you want to remove '%s' and start fresh? (y/N): " % \
                   exp_dir
        answer = raw_input(question)
        if answer.strip() == "y":
            shutil.rmtree(exp_dir)
        else:
            print("Please restart script without -f/--fresh flag")
            sys.exit(0)
    # setup  experiment
    setup_experiment_dir(exp_dir)

    # serialize config for later reference
    save_configuration(config, exp_dir)

    # =====================
    # Define training graph
    # =====================
    with tf.device("/cpu:0"):
        global_step = tf.Variable(starting_step,
                                  name="global_step",
                                  trainable=False)
        train_phase = tf.placeholder(tf.bool)
        learning_rate = setup_learning_rate(config, global_step)
        optimizer = setup_optimizer(config, learning_rate)

        # create input pipeline
        datasets = model.datasets()
        dhandle = tf.placeholder(tf.string)
        iterator = tf.data.Iterator.from_string_handle(
            dhandle,
            datasets['trn'].output_types,
            datasets['trn'].output_shapes
        )
        inputs_full = iterator.get_next()

        # split data across gpus
        inputs_split = split_batch(inputs_full, num_gpus)

        # create model across gpus
        towers = []
        for tidx in range(num_gpus):
            reuse = True if tidx > 0 else None
            inputs = inputs_split[tidx]
            with tf.name_scope("tower_%d" % tidx) as scope, \
                    tf.device("/gpu:%d" % tidx), \
                    tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

                tower = model.build(inputs, train_phase)
                train_vars = filter_trainable_variables(config)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                # setup weight decay
                weight_decay_vars = filter_weight_decay_variables(config)
                tower.loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in
                                          weight_decay_vars],
                                         name="loss_weight_decay") * \
                    weight_decay_value
                tf.summary.scalar("loss_l2", tower.loss_l2)

                # setup total loss
                tower.loss_total = tf.add_n([tower.loss,
                                             tower.loss_l2])

                # compute gradients
                # with tf.control_dependencies(update_ops):
                grads = optimizer.compute_gradients(tower.loss_total,
                                                    train_vars)
                grads = discard_empty_gradients(grads)

                # clip gradients if requested by model
                if clip_gradients_enabled is True:
                    grads = clip_gradients(grads,
                                           minval=clip_gradients_minval,
                                           maxval=clip_gradients_maxval)


                tower.grads = grads
                towers.append(tower)

        # gather, average, and apply gradients
        with tf.name_scope("optimze_step") as scope, \
                tf.device("/gpu:0"):
            loss_avg = tf.add_n(
                [t.loss_total for t in towers]) / float(len(towers))
            grads_avg = average_gradients([t.grads for t in towers])
            train_op = optimizer.apply_gradients(grads_avg,
                                                 global_step=global_step)
            train_op = tf.group(update_ops + [train_op])

        metrics_update_ops = tf.get_collection("metrics_update_ops")
        metrics_update = tf.group(metrics_update_ops)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer(),
                           tf.tables_initializer())

        # setup summaries
        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss_avg)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_data = tf.summary.merge(list(summaries), name='summary_data')

        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=max_checkpoints_to_keep)
        summary_writers = setup_summary_writers(exp_dir)

        # ==================
        # Report model stats
        # ==================

        print("")
        for k, v in towers[0].items():
            if hasattr(v, "get_shape"):
                print("Model definition - shape:%-20s attr:%-30s tensor:%s" %
                      (v.get_shape(), k, v.name))
        print("")

        for op in update_ops:
            print("Update op - %s" % op.name)
        print("")

        for v in train_vars:
            print("Train var - %s" % v.name)
        print("")

        for v in weight_decay_vars:
            print("Weight decay var - %s" % v.name)
        print("")

    # ======================
    # Begin training session
    # ======================
    cfproto = tf.ConfigProto(allow_soft_placement=True)
    G = tf.get_default_graph()
    with tf.Session(graph=G, config=cfproto) as sess:

        trn_ds_handle = get_dataset_handle(datasets['trn'], sess)
        vld_ds_handle = get_dataset_handle(datasets['vld'], sess)

        # --------------------
        # initialize variables
        # --------------------
        sess.run(init_op)

        # -----------------------------------
        # Load checkpoint or pretrained model
        # -----------------------------------

        # from resumed ckpt
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Resuming from latest checkpoint: %s" % latest_checkpoint)
            saver.restore(sess, latest_checkpoint)
            starting_step = sess.run(global_step)
        elif weights_load_path:
            # from manually specified pre-trained weights
            print("Restoring from pre-trained checkpoint: %s" %
                  weights_load_path)
            restore_available(sess, weights_load_path)
        else:
            # from scratch by default
            print("Begin training from scratch")

        # -------------------
        # start training loop
        # -------------------

        # drop into debugging session if specified
        if args.debug:
            import IPython
            IPython.embed()

        try:
            step = None
            start_time = time.time()
            for step in range(starting_step, num_steps):

                # ----------
                # train step
                # ----------
                fetches = {
                    "step": global_step,
                    "train_op": train_op,
                    "loss": loss_avg,
                    "lr": learning_rate,
                    "summary_data": summary_data,
                    "metrics_update": metrics_update
                }
                feed_dict = {
                    dhandle: trn_ds_handle,
                    train_phase: True
                }

                t0 = time.time()
                res = sess.run(fetches, feed_dict)
                step_time = time.time() - t0

                # save summaries
                if (step % summary_save_interval) == 0 or (step == (num_steps - 1)):
                    summary_writers['trn'].add_summary(
                        res['summary_data'], step)

                # print metrics
                if (step % print_interval) == 0 or (step == (num_steps - 1)):
                    print("%s - trn - step:%-6s/%-6s loss:%.5f lr:%.6f secs:%.5f total_mins:%.2f" % (
                        exp_name,
                        step,
                        num_steps,
                        res["loss"],
                        res["lr"],
                        step_time,
                        (time.time() - start_time) / 60.
                    ))

                # stop training if we reach nan
                if np.isnan(res['loss']):
                    break

                # ---------------
                # validation step
                # ---------------
                if (step % validation_step_interval) == 0:
                    fetches = {
                        "step": global_step,
                        "loss": loss_avg,
                        "summary_data": summary_data,
                        "metrics_update": metrics_update
                    }
                    feed_dict = {
                        dhandle: vld_ds_handle,
                        train_phase: False
                    }
                    res = sess.run(fetches, feed_dict)

                    # save summaries & print metrics
                    summary_writers['vld'].add_summary(
                        res['summary_data'], step)
                    print("%s - vld - step:%-6s/%-6s loss:%.5f" % (
                        exp_name,
                        step,
                        num_steps,
                        res["loss"],
                    ))

                # save checkpiont interval
                if (step % checkpoint_save_interval) == 0:
                    if step != 0:
                        print("Saving checkpoint: %s-%s" %
                              (checkpoint_path, step))
                        saver.save(sess, checkpoint_path, global_step=step)

        except KeyboardInterrupt:
            pass

        finally:
            if step:
                # save last checkpoint
                print("\nSaving checkpoint: %s-%s" % (checkpoint_path, step))
                saver.save(sess, checkpoint_path, global_step=step)

            hours_elapsed = (time.time() - start_time) / 3600.0
            print("Training stopped after %.5f hours" % hours_elapsed)
