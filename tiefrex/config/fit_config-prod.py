#!/bin/python3
"""
config file for tiefrex
"""

import os
from tiefrex.constants import SEED, __file__
from tiefrex.data import FeatureSource, InteractionsSource
from tiefrex.constants import FType
from time import strftime, gmtime
from datetime import datetime, timedelta

# Fit Params
emb_dim = 30
batch_size = 1024*16
n_steps = 10000+1
log_every = 100
eval_every = 1000
save_every = 10000

BUCKET_IMPORTS = 's3://cerebro-recengine-imports'
BUCKET_FIXTURES = 's3://cerebro-recengine-fixtures'
BUCKET_EXPORTS = 's3://cerebro-recengine-exports'

TODAY = (datetime.today()-timedelta(2)).strftime('%Y-%m-%d')
YESTERDAY = (datetime.today()-timedelta(3)).strftime('%Y-%m-%d')
n_days_before = lambda n: (datetime.today()-timedelta(n)).strftime('%Y-%m-%d')

train_interactions = InteractionsSource(
    path=os.path.join(
        BUCKET_IMPORTS, 'profile/user-product_activity_counts/'),
    user_col='ops_user_id',
    item_col='ops_product_id',
    activity_col='activity',
    activity_filter_set={
        b'purch',
        b'cart',
        b'list',
    },
    days_lookback=365,
    date_lookforward=n_days_before(3),
)
val_interactions = InteractionsSource(
    path=os.path.join(
        BUCKET_IMPORTS, 'profile/user-product_activity_counts/'),
    user_col='ops_user_id',
    item_col='ops_product_id',
    activity_col='activity',
    activity_filter_set={b'purch'},
    days_lookback=0,
    date_lookforward=n_days_before(2),
)


user_features = []
item_features = [
    FeatureSource(
        name='taxonomy',
        path=os.path.join(BUCKET_IMPORTS,
                          'dim/product_taxonomies.msg'),
        feature_type=FType.CAT,
        index_col='ops_product_id',
    ),
    FeatureSource(
        name='brand',
        path=os.path.join(BUCKET_IMPORTS,
                          'dim/product_brand.msg'),
        feature_type=FType.CAT,
        index_col='ops_product_id',
    ),
    FeatureSource(
        name='price_bucket',
        path=os.path.join(BUCKET_IMPORTS,
                          'features/product_prices.msg'),
        feature_type=FType.CAT,
        index_col='ops_product_id',
    ),
]

val_user_features = user_features  # just re-use
val_item_features = item_features  # just re-use


user_specific_feature = True
item_specific_feature = True

feature_weights_d = {
    'age_bucket': 0.01,
    'gender': 0.01,
    'international_vs_domestic_location': 0.01,
    'loyalty_current_level': 0.01,
}

# More validations params
validation_params = {
    'limit_items': 1000,
    'n_users_eval': 200,
    'include_cold': False,
    'cold_only': False
}

seed = SEED

names = {
#     'ops_brand_id': os.path.join(local_data_dir, 'train/dim/brand_names.csv'),
#     'ops_product_category_id': os.path.join(local_data_dir, 'train/dim/pcat_names.csv'),
}

log_dir = f'/tmp/tensorboard-logs/TR-PROD'

ckpt_upload_s3_uri = os.path.join(BUCKET_EXPORTS, 'tiefrex', 'models-prod')
repr_export_path = os.path.join(BUCKET_EXPORTS, 'tiefrex',
                                'representations-prod')
