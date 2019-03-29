# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

import os
import json
import tensorflow as tf

def json_line_reader(fpath, keys=None, format_fn=None, repeat=True):
    while True:
        with open(fpath) as fh:
            for line in fh:
                try:
                    record = line.split("\t")[0]
                    record = json.loads(record.strip())
                    if format_fn:
                        record = format_fn(record)
                    elif keys:
                        record = dict([(k,v) for k,v in
                                       record.iteritems()
                                       if k in keys])
                    yield record
                except ValueError:
                    continue
        if not repeat:
            break

def json_line_dataset(fpath, output_types, output_shapes,
                      format_fn=None, repeat=True):
    if not os.path.exists(fpath):
        raise IOError("%s does not exist" % fpath)
    keys = output_types.keys() if format_fn is None else None
    gen = json_line_reader(fpath, keys=keys, format_fn=format_fn, repeat=repeat)
    ds = tf.data.Dataset.from_generator(lambda : gen,
                                        output_types=output_types,
                                        output_shapes=output_shapes)
    return ds
