# Trainers

This directory contains trainers that can train models. The main trainer that is
used is the `default.py` trainer that trains a simple supervised model. You can
implement your own trainer, if you need to have more fine-grain control over the
training loop.

For example you might create a new trainer for training GANS since it requires
an alternating optimziation on different losses between two networks.

A trainer is a python module that implements the following methods:

* `train(model, args)`
