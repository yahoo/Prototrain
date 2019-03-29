# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as pth
import json
import random
import numpy as np
import matplotlib.pyplot as plt

DEEP_FASHION_CLASSES_HIGH = ['CLOTHING', 'DRESSES', 'TOPS', 'TROUSERS']
DEEP_FASHION_CLASSES = ['CLOTHING/Blouse', # and button-up shirts
                        'CLOTHING/Coat', # very small
                        'CLOTHING/Jeans', # includes jorts and denim miniskirts
                        'CLOTHING/Pants', # includes shorts. no denim.
                        'CLOTHING/Polo_Shirt', # includes some other things with collar but not button-up
                        'CLOTHING/Summer_Wear', # small, mostly t-shirts
                        'CLOTHING/Tank_Top', # small but seems well-classified
                        'CLOTHING/T_Shirt', # also include some polo shirts
                        'DRESSES/Dress', # lots of variety
                        'DRESSES/Lace_Dress', # very small
                        'DRESSES/Skirt', # includes jean-skirts
                        'DRESSES/Sleeveless_Dress', # very small, unclear why divided from Dress
                        'DRESSES/Suspenders_Skirt', # small but well-classified
                        'TOPS/Blouse', # mostly women's blouses (as opposed to CLOTHING/Blouse)
                        'TOPS/Chiffon', # mostly women's tops of large variety, not always clearly distinct from blouse
                        'TOPS/Coat', # and other outerwear
                        'TOPS/Lace_Shirt', # msotly what it says
                        'TOPS/Summer_Wear', # all over the place, mostly upper-body. even has coats.
                        'TOPS/Tank_Top', # mostly sleeveless, mostly women
                        'TOPS/T_Shirt', # includes some other tops
                        'TROUSERS/Leggings', # and lots of other lower-body pants-like things, even jorts
                        'TROUSERS/Pants', # lower body pants-like things
                        'TROUSERS/Summer_Wear'] # lots of shorts, also some jeans and stuff

DEEP_FASHION_CLASS_MAP = {'CLOTHING/Blouse': 'Misc_Top',
                          'CLOTHING/Coat': 'Coat',
                          'CLOTHING/Jeans': 'Misc_Lower',
                          'CLOTHING/Pants': 'Misc_Lower',
                          'CLOTHING/Polo_Shirt': 'Misc_Top',
                          'CLOTHING/Summer_Wear': 'T_Shirt',
                          'CLOTHING/Tank_Top': 'Sleeveless',
                          'CLOTHING/T_Shirt': 'T_Shirt',
                          'DRESSES/Dress': 'Dress',
                          'DRESSES/Lace_Dress': 'Dress',
                          'DRESSES/Skirt': 'Dress',
                          'DRESSES/Sleeveless_Dress': 'Dress',
                          'DRESSES/Suspenders_Skirt': 'Dress',
                          'TOPS/Blouse': 'Misc_Top',
                          'TOPS/Chiffon': 'Misc_Top',
                          'TOPS/Coat': 'Coat',
                          'TOPS/Lace_Shirt': 'Misc_Top',
                          'TOPS/Summer_Wear': 'Misc_Top',
                          'TOPS/Tank_Top': 'Sleeveless',
                          'TOPS/T_Shirt': 'T_Shirt',
                          'TROUSERS/Leggings': 'Misc_Lower',
                          'TROUSERS/Pants': 'Misc_Lower',
                          'TROUSERS/Summer_Wear': 'Misc_Lower'}

INVERSE_CLASS_MAP = {val: [] for val in DEEP_FASHION_CLASS_MAP.values()}
for key, val in DEEP_FASHION_CLASS_MAP.items():
    INVERSE_CLASS_MAP[val].append(key)

DEEP_FASHION_THREE_CLASS_MAP = {'CLOTHING/Blouse': 'Upper',
                                'CLOTHING/Coat': 'Upper',
                                'CLOTHING/Jeans': 'Lower',
                                'CLOTHING/Pants': 'Lower',
                                'CLOTHING/Polo_Shirt': 'Upper',
                                'CLOTHING/Summer_Wear': 'Upper',
                                'CLOTHING/Tank_Top': 'Upper',
                                'CLOTHING/T_Shirt': 'Upper',
                                'DRESSES/Dress': 'Dress',
                                'DRESSES/Lace_Dress': 'Dress',
                                'DRESSES/Skirt': 'Dress',
                                'DRESSES/Sleeveless_Dress': 'Dress',
                                'DRESSES/Suspenders_Skirt': 'Dress',
                                'TOPS/Blouse': 'Upper',
                                'TOPS/Chiffon': 'Upper',
                                'TOPS/Coat': 'Upper',
                                'TOPS/Lace_Shirt': 'Upper',
                                'TOPS/Summer_Wear': 'Upper',
                                'TOPS/Tank_Top': 'Upper',
                                'TOPS/T_Shirt': 'Upper',
                                'TROUSERS/Leggings': 'Lower',
                                'TROUSERS/Pants': 'Lower',
                                'TROUSERS/Summer_Wear': 'Lower'}

INVERSE_THREE_CLASS_MAP = {val: [] for val in DEEP_FASHION_THREE_CLASS_MAP.values()}
for key, val in DEEP_FASHION_THREE_CLASS_MAP.items():
    INVERSE_THREE_CLASS_MAP[val].append(key)

SOP_CLASSES = ['bicycle', 'cabinet', 'chair', 'coffee_maker', 'fan',
               'kettle', 'lamp', 'mug', 'sofa', 'stapler', 'table', 'toaster']


def topk_accuracy(first_correct, maxk = 50):
    """Returns an array of length maxk where the kth entry is the top-k retrieval accuracy
    if first_correct is an array of the index of the first correct retrieval for each example."""
    total = len(first_correct)
    first_correct = np.array(first_correct)
    return [np.count_nonzero(first_correct <= kk) / total for kk in range(maxk)]

def read_image_list(path):
    part_list = []
    with open(path, 'r') as ff:
        for line in ff:
            part_list.append(line.strip())
    return part_list

def id_from_filename(filename, form='DeepFashion'):
    if form == 'DeepFashion':
        return id_from_filename_df(filename)
    else:
        return id_from_filename_sop(filename)

def id_from_filename_df(filename):
    return 'id_' + filename.split('id_')[1][:8]

def id_from_filename_sop(filename):
    return filename.split('/')[1].split('_')[0]

def class_from_filename(filename, superclass_map=None, form='DeepFashion'):
    if form == 'DeepFashion':
        return class_from_filename_df(filename, superclass_map=superclass_map)
    else:
        return class_from_filename_sop(filename)

def class_from_filename_df(filename, superclass_map=None):
    for class_name in DEEP_FASHION_CLASSES:
        if class_name in filename:
            if superclass_map is not None:
                return superclass_map[class_name]
            return class_name
    raise ValueError('Filename does not contain a DeepFashion class: ' + filename)

def class_from_filename_sop(filename):
    return filename.split('_final')[0]

def class_counts(image_list, print_counts=False, superclass_map=None):
    """Count the number of images in each class."""
    if superclass_map is None:
        class_list = DEEP_FASHION_CLASSES
    else:
        class_list = superclass_map.values()
    count_dict = {key: 0 for key in class_list}
    for img in image_list:
        count_dict[class_from_filename(img, superclass_map)] += 1
    if print_counts:
        for cl, count in count_dict.items():
            print(cl, count)
    return count_dict

def complete_paths(results, newfile, basedir):
    for res_dict in results:
        res_dict['query']['media'] = pth.join(basedir, res_dict['query']['media'])
        for single in res_dict['results']:
            single['media'] = pth.join(basedir, single['media'])
    with open(newfile, 'w') as fh:
        json.dump(results, fh)
    return results

def complete_paths_in_json(oldfile, newfile, basedir):
    with open(oldfile, 'r') as fh:
        results = json.load(fh)
    complete_paths(results, newfile, basedir)

def reduce_json(oldfile, newfile, num_to_keep=100, shuffle=True, sort='rank',
                dataset="DeepFashion"):
    """
    Take num_to_keep entries from the record in oldfile and save to newfile.
    If shuffle, the old records are shuffled before drawing from them.
    If sort is None, the order (after shuffling if applicable) is preserved.
    Otherwise we sort from easiest to hardest by either 'rank'
    (i.e. the index of the first true positive) or by the similarity of the first true positive.
    """
    with open(oldfile, 'r') as fh:
        results = json.load(fh)

    new_results = reduce_record(results, num_to_keep, shuffle, sort, dataset=dataset)

    with open(newfile, 'w') as fh:
        json.dump(list(new_results), fh)

def reduce_record(record, num_to_keep=100, shuffle=True, sort='rank', dataset="DeepFashion"):
    if shuffle:
        random.shuffle(record)

    new_results = []
    for ii in range(min(num_to_keep, len(record))):
        new_results.append(record[ii])

    if sort is not None:
        first_correct = []
        sim_first_correct = []
        for res in new_results:
            correct = get_correct_from_record(res, dataset=dataset)
            first_correct.append(np.argmax(correct))
            try:
                sim_first_correct.append(res['results'][first_correct[-1]]['score'])
            except IndexError:
                sim_first_correct.append(0)
        if sort == 'rank':
            sorter = np.argsort(first_correct)
        else:
            sorter = np.argsort(sim_first_correct)[::-1]
        new_results = np.array(new_results)[sorter]

    return new_results

def get_correct_from_record(res, dataset='DeepFashion'):
    query_id = id_from_filename(res['query']['media'], form=dataset)
    correct = [query_id in retr['media'] for retr in res['results']]
    correct.append(True)
    return correct


def separate_by_class(oldfile, newfile_root, **kwargs):
    with open(oldfile, 'r') as fh:
        results = json.load(fh)

    return separate_by_class_from_record(results, newfile_root, **kwargs)


def separate_by_class_from_record(results, newfile_root, class_list=DEEP_FASHION_CLASSES,
                                  superclass_map=None,
                                  num_to_retrieve=50, reduce=True,
                                  dataset='DeepFashion'):

    first_correct_dict = {key: [] for key in class_list}
    results_by_class = {key: [] for key in class_list}
    for res in results:
        filename = res['query']['media']
        correct = get_correct_from_record(res, dataset=dataset)
        for cl in class_list:
            if cl == class_from_filename(filename, superclass_map,
                                         form=dataset):
                results_by_class[cl].append(res)
                first_correct_dict[cl].append(np.argmax(correct))

    for cl, class_res in results_by_class.items():
        cl_str = cl.replace('/','-')
        if reduce:
            class_res = reduce_record(class_res, dataset=dataset)
        with open(newfile_root+cl_str+'.json', 'w') as fh:
            json.dump(list(class_res), fh)

    ret_at_20 = {key: topk_accuracy(first_correct, num_to_retrieve)[20-1]
                 for key, first_correct in first_correct_dict.items()}

    with open(newfile_root+'ret@20.json', 'w') as fh:
        json.dump(ret_at_20, fh)

    sorter = np.argsort(ret_at_20.values())
    fig = plt.figure(figsize=[7.5,5])
    ax = fig.add_subplot(111)
    ax.bar(np.array(ret_at_20.keys())[sorter],
           np.array(ret_at_20.values())[sorter])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_ylabel('Retrieval accuracy at 20')
    fig.savefig(newfile_root+'_barplot.png', bbox_inches='tight')

    return ret_at_20

def classification_acc_and_confusion(results, superclass_map=None,
                                     base_dir='', experiment_name='ret_as_clf',
                                     dataset='DeepFashion'):
    """Report classification accuracy and confusion matrix using the
    class of the first retrieved item as the class prediction."""
    if superclass_map is not None:
        classes = list(set(superclass_map.values()))
    elif dataset=='DeepFashion':
        classes = DEEP_FASHION_CLASSES
    else:
        classes = SOP_CLASSES
    n_classes = len(classes)
    class_indices = {cl: ii for ii, cl in enumerate(classes)}

    n_correct = 0
    n_predicted = 0
    confusion = np.zeros([n_classes, n_classes])
    for res in results:
        filename = res['query']['media']
        true_class = class_from_filename(filename, superclass_map, form=dataset)
        first_class = class_from_filename(res['results'][0]['media'], superclass_map,
                                          form=dataset)
        n_correct += int(true_class == first_class)
        n_predicted += 1
        confusion[class_indices[true_class],
                  class_indices[first_class]] += 1

    accuracy = n_correct/n_predicted
    print('Classification accuracy at 1: ', accuracy)
    confusion = plot_confusion(confusion, target_names=classes,
                               base_dir=base_dir, experiment_name=experiment_name)
    return accuracy, confusion



def do_all_analysis(exp_dir = None, top_retrieved_file=None, classes='three'):
    """Use exp_dir experiment directory or config.py to find analysis data
    and do the extra analyses in this module."""
    if classes == 'three':
        class_map = DEEP_FASHION_THREE_CLASS_MAP
        class_list = INVERSE_THREE_CLASS_MAP.keys()
    elif classes == 'sop':
        class_map = None
        class_list = SOP_CLASSES

    if exp_dir is None:
        from config import config
        exp_dir = pth.join(config['trainer.experiment_base_dir'],
                           config['trainer.experiment_name'])
    ana_dir = pth.join(exp_dir, 'analysis')
    if top_retrieved_file is None:
        ana_files = os.listdir(ana_dir)
        for af in ana_files:
            if '-top_retrieved.json' in af:
                top_retrieved_file = pth.join(ana_dir, af)

    with open(top_retrieved_file, 'r') as fh:
        results = json.load(fh)

    if classes != 'sop':
        results = complete_paths(results,
                                 top_retrieved_file.replace('top_retrieved',
                                                            'top_retrieved_complete'),
                                 basedir='Deepfashion/img/')

    ret_at_20_by_class = separate_by_class_from_record(results, '', class_list=class_list,
                                                       superclass_map=class_map,
                                                       dataset='SOP' if classes=='sop' else 'DeepFashion')

    reduced = reduce_record(results, dataset='SOP' if classes=='sop' else 'DeepFashion')
    with open(top_retrieved_file.replace('top_retrieved', 'top_retrieved_reduced'), 'w') as fh:
        json.dump(list(reduced), fh)

    _ = classification_acc_and_confusion(results, superclass_map=class_map,
                                         base_dir=exp_dir, experiment_name='',
                                         dataset='SOP' if classes=='sop' else 'DeepFashion')

    return reduced, ret_at_20_by_class
