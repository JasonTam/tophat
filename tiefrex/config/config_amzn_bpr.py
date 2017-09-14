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
    os.path.dirname(os.path.realpath(__file__)), '../data/amazon/')

train_interactions = InteractionsSource(
    path=os.path.join(local_data_dir,
                      # 'splits/tsplit_user-purch_df.msg'
                      # 'splits/tsplit_date_df.msg'
                      'splits/tsplit_100k-women-asin_df.msg'
                      ),
    user_col='reviewerID',
    item_col='asin',
    )

val_interactions = InteractionsSource(
    path=os.path.join(local_data_dir,
                      # 'splits/vsplit_user-purch_df.msg'
                      # 'splits/vsplit_date_df.msg'
                      'splits/vsplit_100k-women-asin_df.msg'
                      ),
    user_col='reviewerID',
    item_col='asin',
    )

item_features = [
]


val_item_features = [
]

user_specific_feature = True
item_specific_feature = True

seed = SEED

names = {
}

batch_size = 1024
log_dir = f'/tmp/tensorboard-logs/amzn_{strftime("%Y-%m-%d-T%H%M%S", gmtime())}'
if not os.path.exists('/tmp/tensorboard-logs/'):
    os.mkdir('/tmp/tensorboard-logs/')
