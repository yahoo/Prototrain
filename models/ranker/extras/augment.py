# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

"""
Image augmentation, cropping, and resizing. Images assumed to be represented
as rgb, each in [0,1].
"""

import tensorflow as tf

def random_noise(image):
    noise = tf.random_normal(tf.shape(image), mean=0, stddev=0.02)
    image = tf.cast(image, tf.float32)
    image = apply_random([
        lambda: noise + image,
        lambda: image
    ])
    return tf.clip_by_value(image, 0.0, 1.0)


def random_crop(image):
    image_shape = tf.shape(image)
    ih = image_shape[0]
    iw = image_shape[1]
    h_percent = tf.random_uniform((), 0.5, 1.0)
    w_percent = tf.random_uniform((), 0.5, 1.0)
    crop_h = tf.cast(tf.cast(ih, tf.float32) * h_percent, tf.int32)
    crop_h = tf.maximum(crop_h, 1)
    crop_w = tf.cast(tf.cast(iw, tf.float32) * w_percent, tf.int32)
    crop_w = tf.maximum(crop_w, 1)
    crop = apply_random([
        lambda: tf.random_crop(image, size=(crop_h, crop_w, 3)),
        lambda: image
    ])
    return crop


def random_resize_method(image, target_size):
    clip = tf.clip_by_value
    image_batch = tf.expand_dims(image, 0)
    image_resized = apply_random([
        lambda: tf.image.resize_bilinear(image_batch,  target_size)[0],
        lambda: tf.image.resize_bicubic(image_batch,  target_size)[0],
        lambda: tf.cast(tf.image.resize_nearest_neighbor(
            image_batch, target_size)[0], tf.float32),
        lambda: tf.image.resize_area(image_batch,  target_size)[0]
    ])
    return image_resized


def random_hflip(image):
    return tf.image.random_flip_left_right(image)


def random_augmentation_and_resize(image, target_size):
    image = random_crop(image)
    image = random_resize_method(image, target_size)
    image = random_hflip(image)
    image = random_noise(image)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image


def apply_random(functions):
    selector = tf.random_uniform(
        (), minval=0, maxval=len(functions), dtype=tf.int32)
    cases = []
    for i in range(len(functions)):
        cases.append((tf.equal(selector, i), functions[i]))
    return tf.case(cases)


def crop_near_bbox(image, bbox, min_cover=0.5):
    """Returns an image cropped and resized, with at least min_cover of the area
    of the given bounding box contained in the crop."""
    image_shape = tf.shape(image)
    ih = image_shape[0]
    iw = image_shape[1]
    bbox = tf.divide(tf.cast(bbox, tf.float32),
                     tf.cast([ih, iw, ih, iw], tf.float32))

    # clip to make sure all bboxes are within bounds
    bbox = tf.clip_by_value(bbox, 0.0, 1.0)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        image_shape,
        bounding_boxes=tf.expand_dims(tf.expand_dims(bbox,0),0),
        min_object_covered=min_cover)
    return tf.slice(image, begin, size)

def crop_to_bbox(image, bbox):
    # image_shape = tf.shape(image)
    # ih = image_shape[0]
    # iw = image_shape[1]
    # bbox = tf.divide(tf.cast(bbox, tf.float32),
    #                  tf.cast([ih, iw, ih, iw], tf.float32))
    return tf.image.crop_to_bounding_box(image, bbox[0], bbox[1],
                                         bbox[2]-bbox[0], bbox[3]-bbox[1])

def crop_to_bbox_and_augment(image, target_size, bbox):
    image = crop_to_bbox(image, bbox)
    image = random_crop(image)
    image = random_resize_method(image, target_size)
    image = random_hflip(image)
    image = random_noise(image)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.)
    return image

def augment_near_bbox_and_resize(image, target_size, bbox):
    image = crop_near_bbox(image, bbox)
    image = random_resize_method(image, target_size)
    image = random_hflip(image)
    image = random_noise(image)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.)
    return image
