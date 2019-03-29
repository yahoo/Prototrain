# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

from commons.config_utils import Config
import os
import os.path as pth

# ************************************************* #
#                   Customize Below                 #
# ************************************************* #

config = Config()


# ================
# Trainer settings
# ================
config.set('trainer.print_interval', 1)
config.set('trainer.validation_step_interval', 100)
config.set('trainer.summary_save_interval', 50)
config.set('trainer.checkpoint_save_interval', 1000)
config.set('trainer.max_checkpoints_to_keep', 100)
config.set('trainer.experiment_base_dir', 'experiments')
config.set('trainer.experiment_name', "exp-huy-ranker-01")
config.set('trainer.random_seed', 1337)

# ================
# Dataset settings
# ================

# uncomment for the dataset you want
# DeepFashion
# ---
# config.set('dataset.name', "DeepFashion")
# config.set('dataset.base_path', pth.join(HOME_DIR, "Data/DeepFashion/"))
# config.set("dataset.image_path", pth.join(HOME_DIR, "Data/DeepFashion/img/"))
# config.set('dataset.domains', True)
# config.set('dataset.intradomain_pairs', True) # ignored for sop
# config.set('dataset.validation', True)        # ignored for DF. for SOP, hold out 10% of train set for validation
# config.set('dataset.bounding_boxes', True)    # ignored if unavailable

# ---

# Stanford_Online_Products
# ---
config.set('dataset.name', "Stanford_Online_Products")
config.set('dataset.base_path', "/path/to/Stanford_Online_Products")
config.set("dataset.image_path", "/path/to/Stanford_Online_Products")
config.set('dataset.domains', False)
config.set('dataset.intradomain_pairs', False)
config.set('dataset.validation', False)        # ignored for DF. for SOP, hold out 10% of train set for validation
config.set('model.random_within_class', .8)
config.set('model.rwc.by_batch', True) # otherwise by epoch
# config.set('model.random_within_class',)
# config.set('model.rwc.by_batch', False) # otherwise by epoch

# ---


# ==============
# Model settings
# ==============
config.set('model.weights_load_path', None)

# triplet loss margin m: L = d(a,p) - d(a,n) + m
config.set('model.margin', 0.1)

# linear dimensionality reduction
config.set('model.reduced_dimension', 512) # eg 128

# specify base image model to use from tfhub
config.set('model.hub_address', "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1")

config.set('model.batch_size', 64)

config.set('model.num_steps', 250000)
config.set('model.default_image_dimensions', [224, 224, 3])

# Select trainable variables
# --------------------------
config.set('model.trainable_variables.filter_fn', )
config.set('model.trainable_variables.filter_regex', )

# Weight decay
# ------------
config.set('model.weight_decay.value', 0.) # 0.00005
# config.set('model.weight_decay.filter_fn', None)
config.set('model.weight_decay.filter_regex', r"((.*weights.*)|(.*kernel.*))")


# Gradient clipping
# -----------------
config.set('model.clip_gradients', False)
config.set('model.clip_gradients.minval', -1.0)
config.set('model.clip_gradients.maxval', 1.0)

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
# config.set('model.optimizer.adam.beta1', 0.9)
# config.set('model.optimizer.adam.beta2', 0.999)
# config.set('model.optimizer.adam.epsilon', 1e-06)

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
# config.set('model.lr_schedule.step.boundaries', [10000, 30000]) # 10k, 30k
# config.set('model.lr_schedule.step.values', [1e-2, 3e-3, 1e-3])
config.set('model.lr_schedule.step.boundaries', [50000])
config.set('model.lr_schedule.step.values', [1e-2, 1e-3])



# Use constant learning rate
# --------------------------
# config.set('model.lr_schedule.method', 'constant')
# config.set('model.lr_schedule.constant.value', 1e-2)

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


# =======================
# Model specific settings
# =======================


# ===============================
# Precision@K Evaluation Settings
# ===============================

# key in model dict for features to use for similarity
config.set("evaluation.precision@k.feature_key", 'query_embeddings')
config.set("evaluation.precision@k.num_to_retrieve", 50)
config.set("evaluation.precision@k.plot", True)
config.set("evaluation.precision@k.max_examples", None) # max number of queries to eval
config.set("evaluation.histogram", True)


# ===============
# Export Settings
# ===============
config.set("actions.export.inputs", ['query'])
config.set("actions.export.outputs", ['query_embeddings'])


# =========
# Overrides
# =========

# add in your runtime configuration here to override any config setting
# based on an environment variable
# if os.environ.get("some_variable"):
#     config.set("foo.bar", 1)
