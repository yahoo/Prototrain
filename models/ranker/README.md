# Training visual search embeddings using triplet loss

This model trains an embedding from a base resnet50 network to be used for
fashion retrieval.

### Setup dependencies

```
pip install 'tensorflow-gpu>=1.70'
pip install tensorflow-hub
```

### Usage

All commands are run from the root of this git repo.

**Getting the training data**

This repository currently supports training on the
[*Stanford Online Products*](http://cvgl.stanford.edu/projects/lifted_struct/)
dataset.

**Training**

```
export CUDA_VISIBLE_DEVICES=0
python train.py -m models.ranker
```

To customize the training hyperparameters, please see config.py. You will likely
need to customize:

```
trainer.experiment_base_dir - change this to a location to store your experimental results
trainer.experiment_name - change this to an experiment name of your liking
trainer.dataset.base_path - path to stanford online product dataset folder
```

**Evaluation**

```
python run.py -a models.ranker.actions.eval -m models.ranker
```

The evaluation results will be stored in `<experiment_dir>/<experiment_name>/analysis`
