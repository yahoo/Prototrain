# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').disabled = True

import sys
import json
import importlib
import tensorflow as tf
from argparse import ArgumentParser

parser = ArgumentParser(usage="Run an action")
parser.add_argument("-m", "--model",
                    help="The model you want to train (ex: 'models.simple')")
parser.add_argument("-a", "--action",
                    help="The action you want to run (ex: 'actions.eval')")
parser.add_argument("-c", "--config",
                    help="JSON string to update config with")
parser.add_argument("-d", "--dataset", default='test',
                    help="Dataset option (e.g., 'trn' for eval on training set")

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

    # update config
    if args.config:
        print("Modifying config with user-specified configuration")
        updates = json.loads(args.config)
        config.update(updates)

    # display config
    print("Loading action from '%s' module" % args.action)
    action = importlib.import_module(args.action)
    action.run(model, args)
