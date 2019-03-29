# Actions

Actions are simply functions that operate on models. This directory contains
actions that are **shared by all models**. You may also implement your own
actions tailored to your specific model by creating a module inside your
 `models/model/actions` folder that implements th following methods:

 * `run(model, args)`

You can call these functions using the `run.py` script:

```
python run.py -a actions.inspect -m models.simple
```
