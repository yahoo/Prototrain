# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

from commons.model_utils import ModelDict
from models.ranker.config import config
from models.ranker.datasets import datasets


def build(inputs=None, train_phase=tf.constant(False)):
    """
    Build network for training and inference
    Args:
    inputs :            (tensor) dict with at least "query" (image shape) and "id" (string)
    train_phase :       (boolean scalar tensor) whether training
    """

    # get config params
    image_dims = config['model.default_image_dimensions']
    hub_address = config['model.hub_address']
    reduced_dimension = config['model.reduced_dimension']
    triplet_margin = config['model.margin']

    model = ModelDict()

    # if no inputs provided via training code, assume we're in inference mode
    # create placeholder and disable training
    if inputs is None:
        inputs = {'query': tf.placeholder(shape=[1] + image_dims,
                                          dtype=tf.float32,
                                          name='input_image_ph')}

    model.query = inputs["query"]
    model.positive = inputs.get("positive")
    model.id = inputs.get("id")

    trainable = "positive" in inputs
    resnet = hub.Module(hub_address,
                        trainable=trainable,
                        tags={"train"} if trainable else None)

    def basenet(images):
        num_features = 2048 if not reduced_dimension else reduced_dimension
        basefeat = resnet(images)
        if reduced_dimension:
            basefeat = tf.layers.dense(basefeat, units=num_features)
            basefeat = tf.identity(basefeat, name='final_reduced')
        basefeat_flat = tf.reshape(basefeat, [-1, num_features])
        basefeat_normed = tf.nn.l2_normalize(basefeat_flat, axis=-1)
        return basefeat_normed

    # setup network for query path
    if "query" in inputs:
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            model.query_embeddings = basenet(model.query)

    # setup network for positive path, add loss
    if "positive" in inputs:
        with tf.variable_scope(scope, reuse=True):
            model.positive_embeddings = basenet(model.positive)

    # setup loss
    if "positive" in inputs:
        model.loss_triplet = triplet_loss(model, margin=triplet_margin)
        model.loss = tf.reduce_mean(model.loss_triplet)
        tf.summary.scalar("loss", model.loss)

    return model


def triplet_loss(model, margin):
    """
    Adds triplet loss to the given model as model.loss. Options and tweaks specified in config.
    """
    query_to_positive, query_to_negative = get_similarities(model)
    loss_triplet = query_to_negative - query_to_positive
    loss_triplet = tf.nn.relu(loss_triplet + tf.constant(margin))

    num_triplets = tf.to_float(tf.size(loss_triplet))
    fraction_not_satisfied =  tf.to_float(tf.count_nonzero(loss_triplet)) / num_triplets
    fraction_hard = tf.to_float(tf.count_nonzero(tf.nn.relu(loss_triplet - margin))) / num_triplets

    tf.summary.scalar('anchor-positive', tf.reduce_mean(query_to_positive))
    tf.summary.scalar('anchor-negative', tf.reduce_mean(query_to_negative))
    tf.summary.scalar('fraction_not_satisfied', fraction_not_satisfied)
    tf.summary.scalar('fraction_hard', fraction_hard)
    return loss_triplet


def get_similarities(model):
    """
    Calculates the similarities required for the comparative loss functions.
    Specify options in config.
    Also declares image summaries (one each of query, positive, negative) for TensorBoard.
    Args:
    model:      ModelDict
    Returns:
    query_to_positive:      (Tensor) similarity measure on query, positive pair
    query_to_negative:      (Tensor) similarity measure on query, negative pair
    """

    query_to_positive, query_to_negative, negative_index = \
        _batch_hard(model.query_embeddings,
                    model.positive_embeddings,
                    model.id)

    model['negative'] = tf.cond(
       tf.greater(negative_index[0],
                  config["model.batch_size"]),
       lambda: tf.gather(model.query,
                         negative_index[0] - config["model.batch_size"]),
       lambda: tf.gather(model.positive,
                         negative_index[0])
    )
    model['negative'] = tf.expand_dims(model['negative'], 0)

    # create image summaries for tensorboard
    tf.summary.image('query_image', model.query, max_outputs=1)
    tf.summary.image('positive_image', model.positive, max_outputs=1)
    tf.summary.image('negative_image', model.negative, max_outputs=1)

    return query_to_positive, query_to_negative


def _batch_hard(query_features, positive_features, item_ids):
    """
    Adapts the 'Batch Hard' triplet mining strategy of Hermans, Beyer, and Leibe.
    This method differs from Batch Hard in that we don't search for the hardest positive
    and just use the one positive example provided.

    Args:
    query_features :     Tensor [batch_size num_features] of normalized features}
    positive_features:   Tensor [batch_size num_features] of normalized features}
    item_ids :           Tensor [batch_size] of item id strings

    Returns:
    positive_sim :       Tensor [batch_size] of query-positive similarities
    hardest_neg... :     Tensor [batch_size] of query-hardest negative similarities
    hardest_neg...index: Tensor [batch_size] of indices (into batch) of hardest negatives
    """

    positive_sim, neg_similarities = _batch_sims(
        query_features, positive_features, item_ids)

    hardest_negative_sim = tf.reduce_max(neg_similarities,
                                         axis=1, keepdims=False)
    hardest_negative_index = tf.argmax(neg_similarities, axis=1)

    return positive_sim, hardest_negative_sim, hardest_negative_index


def _batch_sims(query_features, positive_features, item_ids):
    """
    Computes similarities for all positive and negative pairs in the batch.
    Entries in negative similarities matrix corresponding to positive pairs
    are set to 0.
    """

    dot_products = tf.matmul(query_features,
                             positive_features,
                             transpose_b=True)

    different_id = tf.logical_not(tf.equal(tf.expand_dims(item_ids, 0),
                                           tf.expand_dims(item_ids, 1)))

    neg_similarities = tf.multiply(tf.to_float(different_id), dot_products)

    if config['dataset.intradomain_pairs'] or not config['dataset.domains']:
        query_query = _self_neg_dots(query_features, item_ids)
        neg_similarities = tf.concat([neg_similarities, query_query], axis=1)

    positive_sim = tf.diag_part(dot_products)

    return positive_sim, neg_similarities


def _self_neg_dots(features, item_ids):
    """
    Returns matrix of feature dot products, with products
    corresponding to same-id inputs set to 0.
    """
    dots = tf.matmul(features, features, transpose_b=True)
    different_id = tf.logical_not(tf.equal(tf.expand_dims(item_ids, 0),
                                           tf.expand_dims(item_ids, 1)))
    return tf.multiply(tf.to_float(different_id), dots)
