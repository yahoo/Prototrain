# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

from commons.config_utils import Config

# ************************************************* #
#                   Customize Below                 #
# ************************************************* #

config = Config()

# ================
# Trainer settings
# ================
config.set('trainer.print_interval', 100)
config.set('trainer.validation_step_interval', 100)
config.set('trainer.summary_save_interval', 50)
config.set('trainer.checkpoint_save_interval', 1000)
config.set('trainer.max_checkpoints_to_keep', 5)
config.set('trainer.experiment_base_dir', "experiments")
config.set('trainer.experiment_name', "exp-simple-001")
config.set('trainer.random_seed', 1337)

# ================
# Dataset settings
# ================
config.set('dataset.trn', "data/sample_simple_dataset/trn.txt")
config.set('dataset.vld', "data/sample_simple_dataset/trn.txt")
config.set('dataset.tst', "data/sample_simple_dataset/trn.txt")



# ==============
# Model settings
# ==============
config.set('model.weights_load_path', None)
config.set('model.batch_size', 32)
config.set('model.num_steps', 1000)

# Select trainable variables
# --------------------------
config.set('model.trainable_variables.filter_fn', )
config.set('model.trainable_variables.filter_regex', )

# Weight decay
# ------------
config.set('model.weight_decay.value', 0.00005)
# config.set('model.weight_decay.filter_fn', None)
config.set('model.weight_decay.filter_regex', r"((.*weights.*)|(.*kernel.*))")

# ========================
# Model optimizer settings
# ========================

# Use SGD Momentum optimizer
# --------------------------------
config.set('model.optimizer.method', "msgd")
config.set('model.optimizer.msgd.momentum', .9)
config.set('model.optimizer.msgd.use_nesterov', True)

# Use ADAM optimizer
# ------------------
# config.set('model.optimizer.method', "adam")
# config.set('model.optimizer.adam.beta1', )
# config.set('model.optimizer.adam.beta2',)
# config.set('model.optimizer.adam.epsilon',)

# Use RMSProp optimizer
# ---------------------
# config.set('model.optimizer.method', "rmsprop")
# config.set('model.optimizer.rmsprop.decay')
# config.set('model.optimizer.rmsprop.momentum')
# config.set('model.optimizer.rmsprop.epsilon')


# Use SGD optimizer
# -----------------
# config.set('model.optimizer.method', "sgd")

# ============================
# Model learning rate settings
# ============================

# Use stepwise learning rate
# --------------------------
config.set('model.lr_schedule.method', 'step')
config.set('model.lr_schedule.step.boundaries', [10000, 15000])
config.set('model.lr_schedule.step.values', [2e-3, 1e-3, 1e-4])

# Use constant learning rate
# --------------------------
# config.set('model.lr_schedule.method', 'constant')
# config.set('model.lr_schedule.constant.value', 1e-1)

# Use exponential decay learning rate
# -----------------------------------
# config.set('model.lr_schedule.method', 'exp')
# config.set('model.lr_schedule.exp.value', 1e-3)
# config.set('model.lr_schedule.exp.decay_rate', .95)
# config.set('model.lr_schedule.exp.decay_steps', 10000)

# Use custome learning rate
# -------------------------
# config.set('model.lr_schedule.method', 'custom')
# config.set('model.lr_schedule.custom.fn', lambda step: 1e-3 * .95 ^ (step/100))


# ==================================
# Additional model specific settings
# ==================================

config.set("actions.export.inputs", ["examples"])
config.set("actions.export.outputs", ["probs"])
