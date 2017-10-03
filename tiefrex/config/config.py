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
    os.path.dirname(os.path.realpath(__file__)), '../data/gilt')

BUCKET_IMPORTS = 's3://cerebro-recengine-imports'
BUCKET_FIXTURES = 's3://cerebro-recengine-fixtures'
BUCKET_EXPORTS = 's3://cerebro-recengine-exports'

target_interactions = os.path.join(
    local_data_dir, 'train/profile/user-product_activity_counts')

train_interactions = InteractionsSource(
    path=os.path.join(local_data_dir,
                      'train/profile/user-product_activity_counts'),
    user_col='ops_user_id',
    item_col='ops_product_id',
    activity_col='activity',
    activity_filter_set={
        b'purch',
        # b'cart',
        # b'list',
        # b'click',
        # b'visit',
    }
)

val_interactions = InteractionsSource(
    path=os.path.join(local_data_dir, 'val/profile/user-product_activity_counts'),
    user_col='ops_user_id',
    item_col='ops_product_id',
    activity_col='activity',
    activity_filter_set={b'purch'}
)

user_features = [
    FeatureSource(
        name='user_features',
        path=os.path.join(local_data_dir, 'train/features/user_summary/'),
        feature_type=FType.CAT,
        index_col='ops_user_id',
    ),
]
item_features = [
    FeatureSource(
        name='dim_products',
        path=os.path.join(local_data_dir, 'train/dim/dim_products.msg'),
        feature_type=FType.CAT,
        index_col='ops_product_id',
        # use_cols=['ops_brand_id'],
    ),
    # FeatureSource(
    #     name='product_prices',
    #     path=os.path.join(local_data_dir, 'train/features/product_prices.msg'),
    #     # Note: might want to have this as SCALAR
    #     feature_type=FType.CAT,
    #     index_col='ops_product_id',
    # ),
    # FeatureSource(
    #     name='product_desc',
    #     path=os.path.join(local_data_dir, 'train/features/product_descriptions/ops_product_id-description-D2V.msg'),
    #     feature_type=FType.NUM,
    # ),
]


val_item_features = [  # because features sometimes change and new items are added
    FeatureSource(
        name='dim_products',
        path=os.path.join(local_data_dir, 'val/dim/dim_products.msg'),
        feature_type=FType.CAT,
        index_col='ops_product_id',
        # use_cols=['ops_brand_id'],
    ),
    # FeatureSource(
    #     name='product_desc',
    #     path=os.path.join(local_data_dir, 'train/features/product_descriptions/ops_product_id-description-D2V.msg'),
    #     feature_type=FType.NUM,
    # ),
]

val_user_features = [
    FeatureSource(
        name='user_features',
        path=os.path.join(local_data_dir, 'val/features/user_summary/'),
        feature_type=FType.CAT,
        index_col='ops_user_id',
    ),
]

user_specific_feature = True
item_specific_feature = True

feature_weights_d = {
    'age_bucket': 0.01,
    'gender': 0.01,
    'international_vs_domestic_location': 0.01,
    'loyalty_current_level': 0.01,
}

seed = SEED

names = {
    'ops_brand_id': os.path.join(local_data_dir, 'train/dim/brand_names.csv'),
    'ops_product_category_id': os.path.join(local_data_dir, 'train/dim/pcat_names.csv'),
}

batch_size = 1024
log_dir = f'/tmp/tensorboard-logs/{strftime("%Y-%m-%d-T%H%M%S", gmtime())}'