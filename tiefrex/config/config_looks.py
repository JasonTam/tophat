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

train_interactions = InteractionsSource(
    path=os.path.join(local_data_dir,
                      'looks_may17/train/user-look_activity_counts_purch-cart-list_preloaded.msg'),
                      # 'looks_may17/train/user-look_activity_counts/'),
                      # 'looks_may17/train/purch_looks.msg'),
                      # 'looks_small/train/q.msg'),
    user_col='ops_user_id',
    item_col='ops_product_look_id',
    activity_col='activity',
    activity_filter_set={
        'purch',
        'cart',
        'list',
        # 'click',
        # 'visit',
    }
    )

val_interactions = InteractionsSource(
    path=os.path.join(local_data_dir,
                      'looks_may17/val/user-look_activity_counts/'),
                      # 'looks_may17/val/purch_looks.msg'),
                      # 'looks_small/val/q.msg'),
    user_col='ops_user_id',
    item_col='ops_product_look_id',
    activity_col='activity',
    activity_filter_set={
        'purch',
        # 'cart',
        # 'list',
        # 'click',
        # 'visit',
    }
    )

item_features = [
    FeatureSource(
        name='dim_looks',
        path=os.path.join(local_data_dir,
                          'looks_may17/dim/dim_looks_tax_img-subset.msg'),
                          # 'looks_may17/dim/dim_looks_tax.msg'),
                          # 'looks_small/dim/dim_looks_tax.msg'),
        feature_type=FType.CAT,
        index_col='ops_product_look_id',
        use_cols=[],  # limiting other feature to see if viz feats are helpful
        # use_cols=['ops_product_id'],  # limiting other feature to see if viz feats are helpful
    ),
    # FeatureSource(
    #     name='product_imgs',
    #     path=os.path.join(local_data_dir,
    #                       # 'img_embs_procd/filtered/img_embs.msg'),
    #                       # 'img_embs_procd/filtered/img_embs_scaled.msg'),
    #                       'img_embs_procd/filtered/img_embs_mm_scaled.msg'),
    #     feature_type=FType.NUM,
    # ),
]


val_item_features = [  # because features sometimes change and new items are added
    FeatureSource(
        name='dim_looks',
        path=os.path.join(local_data_dir,
                          'looks_may17/dim/dim_looks_tax_img-subset.msg'),
                          # 'looks_may17/dim/dim_looks_tax.msg'),
                          # 'looks_small/dim/dim_looks_tax.msg'),
        feature_type=FType.CAT,
        index_col='ops_product_look_id',
        use_cols=[],  # limiting other feature to see if viz feats are helpful
        # use_cols=['ops_product_id'],  # limiting other feature to see if viz feats are helpful
    ),
    # FeatureSource(
    #     name='product_imgs',
    #     path=os.path.join(local_data_dir,
    #                       # 'img_embs_procd/filtered/img_embs.msg'),
    #                       # 'img_embs_procd/filtered/img_embs_scaled.msg'),
    #                       'img_embs_procd/filtered/img_embs_mm_scaled.msg'),
    #     feature_type=FType.NUM,
    # ),
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
if not os.path.exists('/tmp/tensorboard-logs/'):
    os.mkdir('/tmp/tensorboard-logs/')
