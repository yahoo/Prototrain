# Copyright 2019, Oath Inc.
# Licensed under the terms of the Apache 2.0 license.
# See LICENSE file in http://github.com/yahoo/prototrain for terms.

from models.ranker import config

def datasets():
    dataset_name = config['dataset.name']

    if dataset_name == "Stanford_Online_Products":
        import stanford_online_products
        return stanford_online_products.datasets()
    elif dataset_name == "DeepFashion":
        import deep_fashion_consumer2shop
        return deep_fashion.datasets()
    elif dataset_name == "cub":
        import cub
        return cub.datasets()
    elif dataset_name == "cars":
        import cars
        return cars.datasets()
    elif dataset_name == "df_inshop":
        import deep_fashion_inshop
        return deep_fashion_inshop.datasets()
