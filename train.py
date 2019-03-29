# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').disabled = True

import sys
import json
import tensorflow as tf
import numpy as np
import importlib
from argparse import ArgumentParser

parser = ArgumentParser(usage="Train a simple supervised neural net")
parser.add_argument("-m", "--model",
                    help="The model you want to train (ex: 'models.simple')")
parser.add_argument("-d", "--debug",
                    action="store_true",
                    help="Launch ipython after model setup")
parser.add_argument("-f", "--fresh",
                    action="store_true",
                    help="Restart experiment from scratch instead of resuming")
parser.add_argument("-t", "--trainer",
                    default="trainers.default",
                    help="The trainer you want to use to train (default: 'trainers.default')")


def print_config(config):
    print("======================")
    print("Configuration Settings")
    print("======================")
    print(config)
    print("=" * 100)
    print("")

if __name__ == '__main__':

    args = parser.parse_args()

    # turn off tf logging
    tf.logging.set_verbosity(tf.logging.ERROR)

    # load model
    print("Loading model from '%s' module" % args.model)
    model = importlib.import_module(args.model)
    config = model.config

    # set random seed before doing anything else
    np.random.seed(config["trainer.random_seed"])
    tf.set_random_seed(config["trainer.random_seed"])

    # display config
    print("Loading trainer from '%s' module" % args.trainer)
    trainer = importlib.import_module(args.trainer)
    trainer.train(model, args)
