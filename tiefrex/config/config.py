#!/bin/python3
"""
config file for tiefrex
"""

import os
from enum import Enum
from tiefrex.constants import SEED, __file__
from typing import NamedTuple
from time import strftime, gmtime


class FeatureType(Enum):
    CATEGORICAL = 1
    CONTINUOUS = 2

FeatureSource = NamedTuple('FeatureSource', [('path', str), ('feature_type',
                                                             FeatureType)])
InteractionsSource = NamedTuple(
    'InteractionsSource', [('path', str), ('user_id_column', str),
                           ('item_id_column', str), ('activity_column', str),
                           ('filter_activity_set', set)])


###                   ###
#  Config Starts Here   #
###                   ###

local_data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../data/gilt')

BUCKET_IMPORTS = 's3://cerebro-recengine-imports'
BUCKET_FIXTURES = 's3://cerebro-recengine-fixtures'
BUCKET_EXPORTS = 's3://cerebro-recengine-exports'

target_interactions = os.path.join(
    local_data_dir, 'train/profile/user-product_activity_counts')

train_interactions = InteractionsSource(
    path=os.path.join(local_data_dir,
                      'train/profile/user-product_activity_counts'),
    user_id_column='ops_user_id',
    item_id_column='ops_product_id',
    activity_column='activity',
    filter_activity_set={b'purch'}
    )

eval_interactions = InteractionsSource(
    path=os.path.join(local_data_dir, 'val/profile/user-product_activity_counts'),
    user_id_column='ops_user_id',
    item_id_column='ops_product_id',
    activity_column='activity',
    filter_activity_set={b'purch'}
    )

# user_features = [
#     FeatureSource(
#         os.path.join(local_data_dir, 'train/features/user_summary/'),
#         FeatureType.CATEGORICAL),
# ]
item_features = [
    FeatureSource(
        os.path.join(local_data_dir, 'train/dim/dim_products.msg'),
        FeatureType.CATEGORICAL),
]


user_specific_feature = True
item_specific_feature = True

seed = SEED

names = {
    'ops_brand_id': os.path.join(local_data_dir, 'train/dim/brand_names.csv'),
    'ops_product_category_id': os.path.join(local_data_dir, 'train/dim/pcat_names.csv'),
}

batch_size = 1024
log_dir = f'/tmp/tensorboard-logs/{strftime("%Y-%m-%d-T%H%M%S", gmtime())}'
