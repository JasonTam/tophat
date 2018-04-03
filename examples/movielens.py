import numpy as np
import pandas as pd
from collections import defaultdict

from tophat.data import FeatureSource, InteractionsSource
from tophat.constants import FType, FGroup
from lightfm.datasets.movielens import fetch_movielens
from jobs.fit_job import FitJob

SEED = 322


def movielens_cfg():
    data = fetch_movielens(
        indicator_features=False,
        genre_features=True,
        min_rating=5.0,  # Pretend 5-star is an implicit 'like'
        download_if_missing=True,
    )

    # Labels for tensorboard projector
    item_lbls_df = pd.DataFrame(data['item_labels']).reset_index()
    item_lbls_df.columns = ['item_id', 'item_lbls']
    genre_lbls_df = pd.DataFrame([l.split(':')[-1]
                                  for l in data['item_feature_labels']]
                                 ).reset_index()
    genre_lbls_df.columns = ['genre_id', 'genre_lbls']

    names = {
        'item_id': item_lbls_df,
        'genre_id': genre_lbls_df,
    }

    # Converting to tophat data containers
    xn_train = InteractionsSource(
        path=pd.DataFrame(np.vstack(data['train'].nonzero()).T,
                          columns=['user_id', 'item_id']),
        user_col='user_id',
        item_col='item_id',
    )

    xn_test = InteractionsSource(
        path=pd.DataFrame(np.vstack(data['test'].nonzero()).T,
                          columns=['user_id', 'item_id']),
        user_col='user_id',
        item_col='item_id',
    )

    genre_df = pd.DataFrame(np.vstack(data['item_features'].nonzero()).T,
                            columns=['item_id', 'genre_id'])

    # Multiple features within the same group not yet supported in tophat
    # keeping only the first genre for each movie :(
    genre_df.drop_duplicates('item_id', keep='first', inplace=True)

    genre_feats = FeatureSource(
        path=genre_df,
        feature_type=FType.CAT,
        index_col='item_id',
        name='genre',
    )

    fit_params = {
        'emb_dim': 30,
        'batch_size': 128,
        'n_steps': 10000+1,
        'log_every': 500,
        'eval_every': 1000,
        'save_every': 10000,

        # 'l2_emb': 1e-5,

        'loss_fn': 'bpr',
        'sample_method': 'uniform',
        'sample_prefetch': 5,
    }

    data_params = {
        'interactions_train': xn_train,
        'interactions_val': xn_test,

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
        'names': names,
    }

    cfg = defaultdict(lambda: None, config_d)

    return cfg


if __name__ == '__main__':
    job = FitJob(fit_config=movielens_cfg())
    job.run()
