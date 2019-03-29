# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os.path as pth
import json

from models.ranker.extras.augment import (
    random_augmentation_and_resize,
    augment_near_bbox_and_resize
)
from models.ranker.config import config

def datasets():
    load_dataset = loader()
    base_path = config['dataset.base_path']

    trn_ds = load_dataset(base_path, "train", shuffle=True, augmentation=True)
    vld_ds = load_dataset(base_path, 'train', shuffle=True, augmentation=False)

    retrieval_query_ds = load_dataset(base_path, 'test',
                                      retrieval="query",
                                      shuffle=False,
                                      augmentation=False,
                                      repeat=False)
    retrieval_index_ds = load_dataset(base_path, 'test',
                                      retrieval="index",
                                      shuffle=False,
                                      augmentation=False,
                                      repeat=False)

    return {"trn": trn_ds,
            "vld": vld_ds,
            "retrieval_query": retrieval_query_ds,
            "retrieval_index": retrieval_index_ds}

def loader():

    def load_dataset(path, part_label, shuffle=False, augmentation=False, retrieval=None, repeat=True):
        """Reads image lists and constructs dataset as specified. Additional options in config.py."""
        # lists of images and the id: files dictionaries are loaded into memory
        id_dict = read_id_dict(pth.join(path, "Ebay_%s.txt" % part_label))
        image_list = []
        for v in id_dict.values():
            for i in v:
                image_list.append(i)


        if shuffle:
            np.random.shuffle(image_list)

        retrieval_kwargs = {"output_types" : {"query": tf.string,
                                              "id": tf.string,
                                              "url": tf.string},
                            "output_shapes" : {"query": tf.TensorShape(None),
                                               "id": tf.TensorShape(None),
                                               "url": tf.TensorShape(None)}}

        if retrieval is not None:
            ds = tf.data.Dataset.from_generator(lambda : singlet_generator(image_list,
                                                                           None,
                                                                           repeat=repeat),
                                                **retrieval_kwargs)
        else:
            ds = load_dataset_pairs(image_list, id_dict, None, repeat=repeat)

        ds = ds.map(lambda r: parse(r, augmentation=augmentation),
                    num_parallel_calls=64)

        batch_size = config['model.batch_size']
        if shuffle:
            ds = ds.shuffle(buffer_size=2*batch_size)

        if retrieval is None:
            ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        else:
            ds = ds.batch(batch_size)
        ds = ds.prefetch(1)
        return ds

    return load_dataset

def parse(path_dict, augmentation=True):
    """
    Replace paths to images with the images themselves, cropped to bounding
    boxes if available and distorted if augmentation (otherwise only resized).
    """
    result = {}
    for key, val in path_dict.items():
        if key+'_bbox' in path_dict:
            result[key] = get_image(val, augmentation=augmentation, bbox=path_dict[key+'_bbox'])
        elif key == 'id' or key == 'url' or key == 'labels' or key == 'category':
            result[key] = val
        elif '_bbox' not in key:
            result[key] = get_image(val, augmentation=augmentation, bbox=None)
    return result

def load_dataset_pairs(image_list, id_dict, bounding_boxes, repeat=True):
    """Returns Dataset of pairs+metadata."""
    output_types = { "query": tf.string,
                     "positive": tf.string,
                     "id": tf.string }
    output_shapes = { "query": tf.TensorShape(None),
                      "positive": tf.TensorShape(None),
                      "id": tf.TensorShape(None)
                    }

    if bounding_boxes is not None:
        output_types["query_bbox"] = tf.int32
        output_types["positive_bbox"] =  tf.int32
        output_shapes["query_bbox"] = tf.TensorShape(None)
        output_shapes["positive_bbox"] = tf.TensorShape(None)


    random_within_class = config['model.random_within_class']
    if config['dataset.intradomain_pairs']:
        id_dict = dict((key, id_dict[key]['consumer'] + id_dict[key]['shop']) for key in id_dict.keys())
        pair_generator = simple_pair_generator
    elif random_within_class is not None:
        pair_generator = simple_rwc_pair_generator
    else:
        pair_generator = simple_pair_generator

    return tf.data.Dataset.from_generator(lambda : pair_generator(image_list,
                                                                  id_dict,
                                                                  bounding_boxes, repeat=repeat),
                                          output_types=output_types,
                                          output_shapes=output_shapes)


def interdomain_pair_generator(image_list_query, id_dict, bounding_boxes, repeat=True):
    """
    We sample query:positive pairs uniformly.
    Generates pairs of paths to jpeg files. Also returns the item id string.
    """
    while True:
        for query in image_list_query:
            item_id = id_from_filename(query)
            item_id_dict = id_dict[item_id]

            for positive in item_id_dict['shop']:
                yield get_pair_dict(query, positive, bounding_boxes)
        if not repeat:
            break

def simple_rwc_pair_generator(_, id_dict, bounding_boxes, repeat=True):
    """
    Randomly in class pair generator for data without domains.
    Currently just supports SOP dataset.
    """
    batch_size = config['model.batch_size']
    keys = id_dict.keys()
    while True:
        np.random.shuffle(keys)
        if config['model.random_within_class'] < np.random.random():
            which_class = 'all'
        else:
            SOP_CLASSES = ['bicycle', 'cabinet', 'chair', 'coffee_maker', 'fan',
                           'kettle', 'lamp', 'mug', 'sofa', 'stapler', 'table', 'toaster']
            which_class = np.random.choice(SOP_CLASSES)
        jj = 0
        for item in keys:
            all_examples = id_dict[item]
            if which_class == 'all' or which_class == class_from_filename(all_examples[0]):
                np.random.shuffle(all_examples)
                for ii in range(int(len(all_examples) / 2)): # if odd num examples, discard one
                    pair =  get_pair_dict(all_examples[2*ii], all_examples[2*ii + 1], bounding_boxes)
                    yield pair
                    jj += 1
                    if jj > batch_size and config['model.rwc.by_batch']:
                        break
            if jj > batch_size and config['model.rwc.by_batch']:
                break
        if not repeat:
            break

def simple_pair_generator(_, id_dict, bounding_boxes, repeat=True):
    """
    We sample all images almost* uniformly, in pairs without regard to query/index distinction.
    Assumes each value in id_dict is just a list of paths.
    *Caveat: ids with odd number of examples will have one randomly chosen example ignored each epoch.
    """
    while True:
        for item in id_dict:
            all_examples = id_dict[item]
            np.random.shuffle(all_examples)
            for ii in range(int(len(all_examples) / 2)): # if odd num examples, discard one
                yield get_pair_dict(all_examples[2*ii], all_examples[2*ii + 1], bounding_boxes)
        if not repeat:
            break

def get_pair_dict(query, positive, bounding_boxes):
    """
    Returns dict with query and positive filenames and metadata.

    query: string
        relative image path to query image
    positive: string
        relative image path to positive image
    bounding_boxes: dict
        a dictionary where the keys are the relative paths to images and values
        are the [x1, y1, x2, y2] bounding box coordinates for object of interest
    """

    pair = {
        'query': complete_path(query),
        'positive': complete_path(positive),
        'id': id_from_filename(query)
    }

    if bounding_boxes is not None:
        pair['query_bbox'] = bounding_boxes["img/"+query]
        pair['positive_bbox'] = bounding_boxes["img/"+positive]

    return pair

def singlet_generator(image_list, bounding_boxes, repeat=True):
    """
    Returns dicts with only query, together with its id and url.
    """
    while True:
        for query in image_list:
            example = {}
            example['query'] = complete_path(query)
            example['id']    = id_from_filename(query)
            example['url']   = query
            if bounding_boxes is not None:
                example['query_bbox'] = bounding_boxes['img/' + query]
            yield example
        if not repeat:
            break


def get_image(filename, augmentation=True, bbox=None):
    """
    Loads data from jpeg file using tf ops, and does cropping
    and/or augmentation as specified.

    Returns Tensor of image.
    """

    jpeg_data = tf.read_file(filename)
    image = tf.image.decode_jpeg(contents=jpeg_data,
                                 channels=3,
                                 name="load_jpeg")
    image = tf.image.convert_image_dtype(image, tf.float32)

    im_dims = config['model.default_image_dimensions']
    th = im_dims[0]
    tw = im_dims[1]
    if bbox is not None:
        bbox = tf.cast(bbox, tf.int32)
        if augmentation:
            image = augment_near_bbox_and_resize(image, [th, tw], bbox)
            image = tf.identity(image, name='augmented_image')
        else:
            image = tf.image.crop_to_bounding_box(image,
                                                  bbox[0],
                                                  bbox[1],
                                                  bbox[2]-bbox[0],
                                                  bbox[3]-bbox[1])
            image = tf.image.resize_area(tf.expand_dims(image, 0),
                                         [th, tw],
                                         name='resize_image')[0]
    else:
        if augmentation:
            image = random_augmentation_and_resize(image, [th, tw])
        else:
            image = tf.image.resize_area(tf.expand_dims(image, 0), [th, tw])[0]

    return image




def load_bounding_boxes(path, switch_order=True):
    """
    Returns a dict {filename: [x1, y1, x2, y2]} given path to
    DeepFashion bbox file. Each x and y are swapped if switch_order.
    """
    bboxes = {}
    with open(path, 'r') as ff:
        num_boxes = ff.readline()
        num_boxes = int(num_boxes)
        fields = ff.readline()
        for ii, line in enumerate(ff):
            imname, clothes_type, source_type, x1, y1, x2, y2 = line.strip().split()
            if switch_order:
                # TensorFlow wants height, width
                box = [y1, x1, y2, x2]
            else:
                box = [x1, y1, x2, y2]
            bboxes[imname] = [int(xx) for xx in box]
    assert len(bboxes) == num_boxes
    return bboxes

def id_from_filename(filename):
    return filename.split('/')[1].split('_')[0]

def read_id_dict(path):
    from collections import defaultdict
    id_dict = defaultdict(list)
    with open(path) as fh:
        fh.readline()
        for line in fh:
            filename = line.strip().split(" ")[-1]
            id = id_from_filename(filename)
            id_dict[id].append(filename)
    return dict(id_dict)

def read_image_list(path):
    part_list = []
    with open(path, 'r') as ff:
        for line in ff:
            part_list.append(line.strip())
    return part_list

def complete_path(filename):
    return pth.join(config['dataset.image_path'], filename)

def class_from_filename(filename):
    return filename.split('_final')[0]
