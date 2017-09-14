#!/bin/python3
"""
config file for tiefrex
"""

import os
from tiefrex.constants import SEED, __file__
from tiefrex.data import FeatureSource, InteractionsSource
from tiefrex.constants import FType
from time import strftime, gmtime


local_data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../data/test/')

train_interactions = InteractionsSource(
    path=os.path.join(local_data_dir,
                      'tsplit_xns.msg'
                      ),
    user_col='user_id',
    item_col='item_id',
    )

val_interactions = InteractionsSource(
    path=os.path.join(local_data_dir,
                      'vsplit_xns.msg'
                      ),
    user_col='user_id',
    item_col='item_id',
    )

item_features = [
    FeatureSource(
        name='item_num_feats',
        path=os.path.join(local_data_dir,
                          'item_num_feats.msg'),
        feature_type=FType.NUM,
    ),
]


val_item_features = [
    FeatureSource(
        name='item_num_feats',
        path=os.path.join(local_data_dir,
                          'item_num_feats.msg'),
        feature_type=FType.NUM,
    ),
]

user_specific_feature = True
item_specific_feature = True

seed = SEED

names = {
}

batch_size = 3
log_dir = f'/tmp/tensorboard-logs/{strftime("%Y-%m-%d-T%H%M%S", gmtime())}'
if not os.path.exists('/tmp/tensorboard-logs/'):
    os.mkdir('/tmp/tensorboard-logs/')
