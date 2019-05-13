# Prototrain

> A training framework for running experiments in computer vision that's based on TensorFlow

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contribute](#contribute)
- [License](#license)


## Background

This project provides a simple framework to experiment with and develop machine learning models.

More specifically, at Verizon Media we have used this code to perform research on the topic
of "visual search" and "metric learning" within the e-commerce domain. With this repository,
you can train image retrieval models to find relevant products from a catalog of thousands of products
given a single query image.

We have presented this work in a poster session at the
[BayLearn](http://www.baylearn.org/) Symposium on Oct 11, 2018.
To learn more about our work, please read our [technical report](https://arxiv.org/abs/1810.04652)
"Learning embeddings for Product Visual Search with Triplet Loss and Online Sampling".

The training code in this repository aims to provide a flexible framework for
experimenting with low-level neural network details. While high-level APIs such
as tf.estimator and tf.keras provide a quick and simple way of training standard
convolutional neural networks, they abstract away many details of the training
loop that could be interesting to examine within a research context.

The goal of this code base is to give researchers finer-grain controls over these
details while still providing "just enough" infrastructure to track, configure, and
reproduce experiments in a collaborative setting.

A few benefits of using this code for training:

* __Simple Model definition API__ - All models can be created by defining a `model.config` dictionary object and the following two functions: `model.datasets()` and `model.build(inputs, train_phase)`.
* __Debug mode__ - Use the `--debug` flag when calling train.py to jump into an IPython session immediately after model setup has completed.
* __Resumable models__ - You can interrupt and restart the training job without worry. The trainer will pickup from the last saved checkpoint. If you want to start fresh, use the `--fresh` flag to delete the existing  checkpoint and start from scratch.
* __Checkpoint on exit__ - If your job exits for any reason (including if you manually interrupt training), the model will be saved to a checkpoint before exiting.
* __Config serialization__ - All training runs will save the `config` hyper-parameter dictionary as both a human readable text as well as a machine readable format for later reference.
* __Transparent/Customizable training loop__ - The training loop is easy to understand to make it simple to use. You can customize your training loop by copying `trainers/default.py` into a new file.
* __Hyperparameter tuning__ - Provides the possibility for automated hyper-parameter tuning since `config` is nothing more than a python dictionary that may be manipulated at runtime.
* __Multi-GPU Training__ - Train on multiple GPUs at a time.


## Install

This code currently runs in Python 2.7.11, so make sure you have this version running. We have plans to migrate to Python 3 in the future.

Before using this library, you will need to have a few prerequisates installed via pip:

``` pip install numpy tensorflow tensorflow-hub```

Then, simply clone this repo.

```
git clone https://github.com/yahoo/prototrain && cd prototrain
```

## Usage

You can start a new experiment by calling the following code from the command line.

```
export CUDA_VISIBLE_DEVICES=1,2
python train.py -m models.simple
```

This will train the model in the `models/simple` folder. You

## Configuration

The following are configuration flags shared across all models.

### Trainer settings

| dtype | setting | description |
| --------| ------| ----------- |
| `str` | ` trainer.experiment_base_dir` |  The directory path to store all experiments |
| `str` | ` trainer.experiment_name` |  The experiment name. This, combined with `experiment_base_dir`, will be where all experiment logs, checkpoints, and summaries will be saved. The suggested format for this string is `"exp-<initials>-<descriptive_name>-<run_number>"` like the following example: `"exp-hn-evergreen-pascal-voc80"` |
| `int` | ` trainer.validation_step_interval` |  Determines interval of steps between feeding a batch of validation data through session |
| `int` | ` trainer.summary_save_interval` |  Determines interval of steps between saving tensorboard summaries |
| `int` | ` trainer.print_interval` |  Determines interval of steps between printing loss |
| `int` | ` trainer.checkpoint_save_interval` |  Determines interval of steps between saving checkpoints |
| `int` | ` trainer.max_checkpoints_to_keep` |  The number of checkpoints to save at any given time |
| `str` | ` trainer.random_seed` |  A random seed to initialize tensorflow and numpy with |


### Model general settings


| dtype | setting | description |
| --------| ------| ----------- |
| `str`   | `model.weights_load_path` | A path to a pretrained checkpoint that you'd like to load for fine tuning |
| `int`   | `model.batch_size` | The full batch size to use for training |
| `int`   | `model.num_steps` | The number of steps to run trianing |
| `func`  | `model.trainable_variables.filter_fn` | A function of the form `fn(v) -> Bool` that returns `True` if variable `v` should be updated via gradient descent during training |
| `str`   | `model.trainable_variables.filter_regex` | A regex that returns a match if a variable should be updated during training |
| `func`  | `model.weight_decay.filter_fn` | A function of the form `fn(v) -> Bool` that returns `True` if variable `v` should be considered when computing the weight decay loss |
| `str`   | `model.weight_decay.filter_regex` | A regex that returns a match if a variable should be considered when computing the weight decay loss |
| `bool`  | `model.clip_gradients` | Turn on gradient clipping during training (default: False)|
| `float` | `model.clip_gradients.minval` | Minimum gradient value (default: -1.0) |
| `float` | `model.clip_gradients.maxval` | Maximum gradient value (default: 1.0)  |

### Model optimizer settings

| dtype | setting | description |
| --------| ------| ----------- |
| `str` | `model.optimizer.method` | Choose one of the following optimizers `msgd` (Momentum SGD), `adam`, `rmsprop`, `sgd`. Based upon this value, additional settings apply below. |

Momentum SGD settings:

| dtype | setting | description |
| --------| ------| ----------- |
| `float` |`model.optimizer.msgd.momentum` | Sets momentum when using `msgd` optimizer |
| `bool` |`model.optimizer.msgd.use_nesterov` | Uses nesterov accelerated gradients if `True` |

ADAM settings:

| dtype | setting | description |
| --------| ------| ----------- |
| `float` |`model.optimizer.adam.beta1` |  ADAM beta1 |
| `float` |`model.optimizer.adam.beta2` |  ADAM beta2 |
| `float` |`model.optimizer.adam.epsilon` |  ADAM epsilon |

RMSProp settings:

| dtype | setting | description |
| --------| ------| ----------- |
| `float` | `model.optimizer.rmsprop.decay` |  RMSProp decay rate |
| `float` | `model.optimizer.rmsprop.momentum` |  RMSProp momentum |
| `float` | `model.optimizer.rmsprop.epsilon` |  RMSProp epsilon |


### Model learning rate schedule

| dtype | setting | description |
| --------| ------| ----------- |
|  `str`  | `model.lr_schedule.method` | One of the following [`step`, `exp`, `constant`, `custom`]. Based upon this value, additional settings apply below.

Piecewise learning rate schedule:

| dtype | setting | description |
| --------| ------| ----------- |
|  `list(int)` | `model.lr_schedule.step.boundaries` | A list of step boundary values that determine when to set the learning rate to the next learning rate value. |
|  `list(float)` | `model.lr_schedule.step.values` | List of floating point values for the learning rate. This list should have `len(boundaries) + 1` entries, where the first entry is the starting learning rate. |

Constant learning rate schedule:

| dtype | setting | description |
| --------| ------| ----------- |
|  `float` | `model.lr_schedule.constant.value` | A constant value to be used for the learning rate throughout the entire training job. |


Exponential decay learning rate:

| dtype | setting | description |
| --------| ------| ----------- |
| `float` | `model.lr_schedule.exp.value` |  Initial value for learning rate |
| `float` | `model.lr_schedule.exp.decay_rate` | A value to decay learning rate by |
| `float` | `model.lr_schedule.exp.decay_steps` | Number of steps to run before decaying the learning rate |

Custom learning rate schedule:

| dtype | setting | description |
| --------| ------| ----------- |
| `func`  |`model.lr_schedule.custom.fn` | A function of the form `fn(step) -> tf.float` that returns a float value that represents the learning rate at the provided `step`. |


## Contribute

Please refer to [the contributing.md file](Contributing.md) for information about how to get involved. We welcome issues, questions, and pull requests. Pull Requests are welcome.

## Core contributors / Maintainers
Eric Dodds: eric.mcvoy.dodds@verizonmedia.com

## Acknowledgements

We thank the following people for their contributions, testing, and feedback on this code:

* Huy Nguyen
* Jack Culpepper
* Simao Herdade
* Kofi Boakye
* Andrew Kae
* Tobias Baumgartner
* Armin Kappeler
* Pierre Garrigues

## License
This project is licensed under the terms of the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) open source license. Please refer to [LICENSE](LICENSE) for the full terms.
