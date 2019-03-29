# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

import tensorflow as tf

def preprocess(images, method="tensorflow_vgg"):
    """
    Setup image preprocessing function

    Args:
        preprocessing_name: string
            Choose "caffe_vgg", "tensorflow_vgg", or "tensorflow_inception"
            styles of preprocessing. Further details as to which to
            choose can be found here http://bit.ly/2z6jiKa
        images: tf.uint8 or tf.float32 [N, H, W, C]
            Your batch of images array in RGB format
    Returns:
        Your images preprocessed
    """
    images = tf.to_float(images)
    rgb_mean = tf.constant([123., 117., 104.], dtype=images.dtype)
    if method == "caffe_vgg":
        images = images - rgb_mean
        images = images[:, :, :, ::-1]  # swap rgb -> bgr
    elif method == "tensorflow_vgg":
        images = images - rgb_mean
    elif method == "tensorflow_inception":
        images = 2. * (images/255. - 0.5)
    else:
        images = 2. * (images/255. - 0.5)
    return images
