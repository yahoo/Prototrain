# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf
import json
from collections import OrderedDict
from commons.checkpoint_utils import restore_available
from models.ranker.extras.utils import read_image_list, topk_accuracy


def compute_features(model,
                     datasets,
                     checkpoint_dir,
                     weights_load_path,
                     feature_key,
                     dataset_key='retrieval_index'):
    """
    Calculate all features for the index set (or for query set if query_set).
    """
    iterator = datasets[dataset_key].make_one_shot_iterator()
    inputs = iterator.get_next(name='index_set_iterator')
    mdict = model.build(inputs)
    init_op = tf.global_variables_initializer()

    catalog_features = []
    catalog_item_ids = []
    catalog_urls = []
    with tf.Session() as sess:
        sess.run(init_op)
        _initialize(sess, checkpoint_dir, weights_load_path)

        batch_size = model.config["model.batch_size"]
        batches_finished = 0
        while True:
            try:
                fetches = []
                fetches.append(mdict[feature_key])
                fetches.append(mdict["id"])
                if "url" in inputs:
                    fetches.append(inputs["url"])

                bottleneck_batch, ids_batch, urls_batch = sess.run(fetches)

                catalog_features.append(bottleneck_batch)
                catalog_item_ids.append(ids_batch)
                catalog_urls.append(urls_batch)

                batches_finished += 1
                if batches_finished % 30 == 0:
                    print ("Calculating index set features, finished: ",
                           batches_finished * batch_size)
            except tf.errors.OutOfRangeError:
                break
    tf.reset_default_graph()
    catalog_features = np.concatenate(catalog_features, axis=0)
    catalog_item_ids = np.concatenate(catalog_item_ids, axis=0)
    catalog_urls = np.concatenate(catalog_urls, axis=0)
    return catalog_features, catalog_item_ids, catalog_urls


def _initialize(sess, checkpoint_dir, weights_load_path):
    """Load model."""

    # load model from latest checkpoint if one available
    # load model from specified checkpiont if one specified
    if weights_load_path:
        print("Restoring from pre-trained checkpoint: %s" %
              weights_load_path)
        saver = tf.train.Saver()
        restore_available(sess, weights_load_path)
    else:
        print("Searching '%s' for latest checkpoints ... " % checkpoint_dir)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Resuming from latest checkpoint: %s" % latest_checkpoint)
            saver = tf.train.Saver()
            restore_available(sess, latest_checkpoint)
        # run eval from scratch by default
        else:
            print("Begin evaluating from untrained model")


def mkdirs(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        pass


def run(model, args):
    """
    model : a module with "build" and "datasets" functions

    The model dictionary returned by "build" should have a key specified by
    config['evaluation.precision@k.feature_key'],
    of which the value is the tensor to be used as features for the similarity retrieval.

    The datasets function should return at least two datasets under keys "retrieval_query"
    and "retrieval_index".
    The retrieval datasets should have two keys:
    "query" : the image; this is a bit of a misnomer since it doesn't have to be a query image
    "query_id" : the item id as a string "id_########"
    """
    config = model.config

    # build datasets
    datasets = model.datasets()
    path = model.config['dataset.base_path']

    print ('Evaluating on test set.')
    query_set_key = 'retrieval_query'
    index_set_key = 'retrieval_index'
    an_dir_prefix = ''

    feature_key = config['evaluation.precision@k.feature_key']
    num_to_retrieve = config['evaluation.precision@k.num_to_retrieve']
    plot = config['evaluation.precision@k.plot']
    max_examples = config['evaluation.precision@k.max_examples']

    experiment_base_dir = config['trainer.experiment_base_dir']
    experiment_name = config['trainer.experiment_name']
    exp_dir = os.path.join(experiment_base_dir, experiment_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoint")
    weights_load_path = config['model.weights_load_path']
    analysis_dir = os.path.join(exp_dir, an_dir_prefix + "analysis")

    if weights_load_path is None or weights_load_path == "":
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            checkpoint_name = "untrained"
        else:
            checkpoint_name = os.path.split(latest_checkpoint)[-1]
    else:
        checkpoint_name = os.path.split(weights_load_path)[-1]

    evaluation_res_file = os.path.join(analysis_dir,
                                       "%s-eval.json" % checkpoint_name)
    evaluation_plot = os.path.join(analysis_dir,
                                   "%s-precision_at_k.png" % checkpoint_name)
    top_retrieved_file = os.path.join(analysis_dir,
                                      "%s-top_retrieved.json" % checkpoint_name)
    histogram_file = os.path.join(analysis_dir,
                                  "%s-histogram.png" % checkpoint_name)

    # setup analysis folder
    mkdirs(analysis_dir)

    catalog_features, catalog_ids, catalog_urls = compute_features(
        model, datasets,
        checkpoint_dir,
        weights_load_path,
        feature_key,
        dataset_key=index_set_key
    )
    train_phase = tf.constant(False)
    iterator = datasets[query_set_key].make_one_shot_iterator()
    inputs = iterator.get_next()
    mdict = model.build(inputs, train_phase)

    # add node to calculate query's similarity with retrieval set via placeholder
    # these are assumed to be normalized outputs from the penultimate layer.
    retrieval_ph = tf.placeholder(shape=[None, mdict[feature_key].shape[-1]],
                                  dtype=tf.float32)
    similarities = tf.matmul(
        mdict[feature_key], retrieval_ph, transpose_b=True)

    # display model structure
    print("")
    for k, v in mdict.items():
        if hasattr(v, "get_shape"):
            print("Model definition - shape:%-20s attr:%-30s tensor:%s" %
                  (v.get_shape(), k, v.name))
    print("")

    if config['evaluation.histogram']:
        bin_bounds = np.arange(0, 1.01, 0.01)
        positive_histo = np.zeros(len(bin_bounds) - 1)
        negative_histo = np.zeros_like(positive_histo)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init_op)
        _initialize(sess, checkpoint_dir, weights_load_path)

        first_correct = []
        results_dicts = []
        while True:
            list_type_catalog_urls = list(catalog_urls)
            try:
                res = sess.run({"similarities": similarities,
                                "query_id": inputs['id'],
                                "url": inputs['url']},
                                feed_dict={retrieval_ph: catalog_features})
                for sims, item_id, url in zip(res["similarities"], res["query_id"], res['url']):
                    # get order of similarities, and top retrieved list
                    # if index and query set are the same, we need to remove retrieval of query itself
                    try:
                        ind_in_catalog_ids = list_type_catalog_urls.index(url)
                    except ValueError:
                        pass
                    else:
                        sims[ind_in_catalog_ids] = 0
                    order = np.argsort(sims)[::-1]  # most similar first
                    top_retrieved = [catalog_ids[order[ii]]
                                     for ii in range(num_to_retrieve)]

                    # get index for first correct retrieval
                    correct = [item_id in match for match in top_retrieved]
                    # so argmax won't return 0 if nothing is correct
                    correct.append(True)
                    first_correct.append(np.argmax(correct))

                    retrieved_urls = [catalog_urls[order[ii]]
                                      for ii in range(num_to_retrieve)]
                    top_sims = [float(sims[order[ii]])
                                for ii in range(num_to_retrieve)]
                    results_dicts.append(get_results_dict(
                        url, retrieved_urls, top_sims))

                    if config['evaluation.histogram']:
                        all_sims = np.array(sims)
                        match_mask = np.array(
                            [item_id in match for match in catalog_urls])
                        pos_freqs, _ = np.histogram(
                            all_sims[match_mask], bins=bin_bounds)
                        neg_freqs, _ = np.histogram(
                            all_sims[~match_mask], bins=bin_bounds)
                        positive_histo += pos_freqs
                        negative_histo += neg_freqs

                num_done = len(first_correct)
                if num_done % 1024 == 0:
                    print (num_done)
                if max_examples is not None and num_done > max_examples:
                    break

            except tf.errors.OutOfRangeError:
                break

    if config['evaluation.histogram']:
        make_histogram(bin_bounds, positive_histo,
                       negative_histo, histogram_file)

    with open(top_retrieved_file, "w") as fh:
        json.dump(results_dicts, fh)

    topk = topk_accuracy(first_correct, maxk=num_to_retrieve)
    print("Experiment logs: %s" % exp_dir)
    print("recall@k -- k=1:%.4f k=10:%.4f k=20:%.4f" % (topk[0], topk[10-1], topk[20 - 1]))

    if plot:
        fig = plt.figure(figsize=[8, 6])
        ax = fig.add_subplot(111)
        line, = ax.plot(topk)
        #ax.legend([line], ['ResNet50 + cosine sim'])
        ax.set_xlabel('Retrieved images (k)')
        ax.set_ylabel('Retrieval accuracy')
        fig.savefig(evaluation_plot)

    with open(evaluation_res_file, "w") as fh:
        json.dump({"precision@k": topk}, fh)

    return topk


def get_results_dict(url, retrieved, top_sims):
    return {"query": {"media": url},
            "results": [{"media": ret, "score": sim}
                        for ret, sim in zip(retrieved, top_sims)]}


def make_histogram(bin_bounds, positive_histo, negative_histo, histogram_file):
    """
    Make and save (and save data for) histogram with given bins.
    Positive and negative are plotted on independent axes but overlaid.
    """
    fig = plt.figure(figsize=[8, 6])
    ax_pos = fig.add_subplot(111)
    ax_pos.set_xlabel('Similarity')
    ax_neg = ax_pos.twinx()
    ax_pos.bar(bin_bounds[:-1], positive_histo, width=bin_bounds[1] - bin_bounds[0],
               align='edge', color='g', alpha=0.5)
    ax_neg.bar(bin_bounds[:-1], negative_histo, width=bin_bounds[1] - bin_bounds[0],
               align='edge', color='r', alpha=0.5)
    ax_pos.set_ylabel("Positive Count", color='g')
    ax_neg.set_ylabel("Negative Count", color='r')
    ax_pos.tick_params('y', colors='g')
    ax_neg.tick_params('y', colors='r')
    plt.tight_layout()
    fig.savefig(histogram_file, bbox_inches='tight')
    with open(histogram_file.replace('png', 'json'), 'w') as fh:
        json.dump({'bin_bounds': list(bin_bounds),
                   'positive_freq': list(positive_histo),
                   'negative_freq': list(negative_histo)}, fh)
