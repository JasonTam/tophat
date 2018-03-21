import numpy as np
import pandas as pd
import os
from collections import defaultdict

from tophat.data import FeatureSource, InteractionsSource
from tophat.constants import FType, FGroup
from lightfm.datasets.movielens import fetch_movielens
from jobs.fit_job import FitJob

SEED = 322

# Outer scope shared params
data = fetch_movielens(
    indicator_features=False,
    genre_features=True,
    min_rating=5.0,  # Pretend 5-star is an implicit 'like'
    download_if_missing=True,
)

# Converting to tophat data containers
xn_train = InteractionsSource(
    path='via_lightfm.datasets.movielens',
    user_col='user_id',
    item_col='item_id',
)
xn_train.data = pd.DataFrame(np.vstack(data['train'].nonzero()).T,
                             columns=['user_id', 'item_id'])

xn_test = InteractionsSource(
    path='via_lightfm.datasets.movielens',
    user_col='user_id',
    item_col='item_id',
)
xn_test.data = pd.DataFrame(np.vstack(data['test'].nonzero()).T,
                            columns=['user_id', 'item_id'])

genre_feats = FeatureSource(
    name='genre',
    path='via_lightfm.datasets.movielens',
    feature_type=FType.CAT,
    index_col='item_id',
)
genre_feats.data = pd.DataFrame(np.vstack(data['item_features'].nonzero()).T,
                                columns=['item_id', 'genre_id'])
genre_feats.data.set_index('item_id', inplace=True)

# Multiple features within the same group not yet supported in tophat
# keeping only the first genre for each movie :(
genre_feats.data = genre_feats.data \
    .loc[~genre_feats.data.index.duplicated(keep='first')]

# Synthetic Genre Favorites Interactions
genre_favs = InteractionsSource(
    path='genre_fav',
    user_col='user_id',
    item_col='genre_id',
)
genre_favs.data = pd.read_msgpack(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 '../data/movielens', 'genre_favs_synthetic.msg')
)


def movielens_cfg():

    fit_params = {
        'emb_dim': 30,
        'batch_size': 128,
        'n_steps': 10000+1,
        'log_every': 500,
        'eval_every': 1000,
        'save_every': 99999,

        # 'l2_emb': 1e-5,

        'loss_fn': 'bpr',
        'sample_method': 'uniform',
        'sample_prefetch': 5,
    }

    data_params = {
        'interactions_train': xn_train,
        'interactions_val': xn_test,

        'interactions_aux': [
            genre_favs,
        ],

        'group_features': {
            FGroup.USER: [],
            FGroup.ITEM: [genre_feats],
        },

        'val_group_features': {
            FGroup.USER: [],
            FGroup.ITEM: [genre_feats],
        },

        'specific_feature': {
            FGroup.USER: True,
            FGroup.ITEM: True,
        },
    }

    validation_params = {
        'limit_items': -1,
        'n_users_eval': 100,
        'include_cold': False,
        'cold_only': False
    }

    config_d = {
        **fit_params,
        **data_params,
        'validation_params': validation_params,
        'seed': SEED,
        'log_dir': f'/tmp/tensorboard-logs/tophat-movielens',
    }

    cfg = defaultdict(lambda: None, config_d)

    return cfg


if __name__ == '__main__':
    job = FitJob(fit_config=movielens_cfg())
    job.run()
