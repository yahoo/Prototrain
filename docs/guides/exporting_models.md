# How to export models

You can export your models as a Tensorflow SavedModel by calling the following
*action*:

```
python run.py -a actions.export -m models.simple
```

Where `models.simple` should be replaced with your desired model. To use this
*action*, you must define the following config keys in your `model.config`:

    "actions.export.inputs":
        A list of tensors from your model dict to use as inputs.

        example:
            ["images"]

    "actions.export.outputs":
        A list of tensors from your model dict to use as outputs.

        example:
            ["pred_probs"]

    "trainer.experiment_base_dir":
        The base directory for all your experiments

        example:
            "/home/foobar/experiments"

    "trainer.experiment_name"
        Your experiment name from which to look for the latest checkpoint

        example:
            "exp-hn-foobar-classifier-001"


### FAQs

##### What is happening underneath the hood?

By default, this action will first call your `model.build()` function,
then instantiate a `tf.Session`, and finally load your latest trained checkpoint
before saving out a SavedModel.

##### Where does the model get saved?

By default, the action will save your model into your experiment directory under
`<experiment_base_dir>/<experiment_name>/exported_model`.

##### How should I set the batch_size of my inputs to be a given size?

Setting your input batch size to `None` is often desirable when serving models in
frameworks like Tensorflow Serving where data may be dynamically batched.

This export *action* does not control the batch size, but rather relies on
your `model.build` function to determine this setting. As a result, you should
write your `model.build` function such that when it is invoked without any arguments,
it generates an "inference" graph with `batch_size` set to your desired setting.

##### What if I want more customized behavior?

You can always customize the behavior of `actions/export.py` by copying it to a new file
and modifying it to do what you want. If for example you created your own export action in
`models/my_model/actions/my_export.py`. You could invoke it like so:

```
python run.py -a models.my_model.actions.my_export -m models.my_model
```
