# How to perform hyper parameter grid search

When training deep models it is often desirable to be able to train and evaluate
with a variety of hyper parameters. Changing the configuration file for every single
hyper-parameter setting can quickly become tedious.

Since each model's configuration is determined by its `config.py`
, which can execute ordinary python code, you can use an environment
variable through `os.environ` to dynamically change the settings of your model.

For each new setting, you would need to change the experiment name so that
results will get saved to separate folder.

## Example code

The code below can be added to the end of your `config.py` to search over a
grid of hyper-parameters:

```python
# ==========================
# Hyperparameter grid search
# ==========================

import itertools as it

# Define a hyper parameter grid to search over.
# keys refer to the configuration setting you'd like to set,
# and values refer to a list of possible values for that configuration setting
param_grid = {
    'model.weight_decay.value': [0.0, 0.00001, 0.0001],
    'model.batch_size': [50, 32, 128]
}
param_names, param_choices = zip(*param_grid.items())
param_combinations = list(it.product(*param_choices))
param_index = int(os.environ['PARAM_INDEX'])
param_values = param_combinations[param_index]

# iterate over chosen parameter combination and set configuration
for key, value in zip(param_names, param_values):
    config[key] = value

# change experiment name accordingly
new_experiment_name = config['trainer.experiment_name'] + "-%d" % param_index
config['trainer.experiment_name'] = new_experiment_name
```

You can then use this code by running the following shell script for training

```shell
for i in {0..9}
do
    export PARAM_INDEX=$i
    python train.py -m models.simple
done
```
