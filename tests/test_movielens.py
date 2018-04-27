import pytest
import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
import tophat.callbacks as cbks
from tophat.data import FeatureSource, InteractionsSource
from tophat.constants import FType, FGroup
from tophat.tasks.wrapper import FactorizationTaskWrapper
from tophat.core import TophatModel
from tophat.evaluation import Validator

from lightfm.datasets.movielens import fetch_movielens


@pytest.fixture
def data():
    # Get movielens data via lightfm
    movielens = fetch_movielens(
        indicator_features=False,
        genre_features=True,
        min_rating=5.0,  # Pretend 5-star is an implicit 'like'
        download_if_missing=True,
    )

    # Labels for tensorboard projector
    item_lbls_df = pd.DataFrame(movielens['item_labels']).reset_index()
    item_lbls_df.columns = ['item_id', 'item_lbls']
    genre_lbls_df = pd.DataFrame([l.split(':')[-1]
                                  for l in movielens['item_feature_labels']]
                                 ).reset_index()
    genre_lbls_df.columns = ['genre_id', 'genre_lbls']

    names_d = {
        'item_id': item_lbls_df,
        'genre_id': genre_lbls_df,
    }

    # #################### [ INTERACTIONS ] ####################
    xn_train = InteractionsSource(
        path=pd.DataFrame(np.vstack(movielens['train'].nonzero()).T,
                          columns=['user_id', 'item_id']),
        user_col='user_id',
        item_col='item_id',
    )

    xn_test = InteractionsSource(
        path=pd.DataFrame(np.vstack(movielens['test'].nonzero()).T,
                          columns=['user_id', 'item_id']),
        user_col='user_id',
        item_col='item_id',
    )

    # #################### [ FEATURES ] ####################

    genre_df = pd.DataFrame(np.vstack(movielens['item_features'].nonzero()).T,
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

    primary_group_features = {
        FGroup.USER: [],
        FGroup.ITEM: [genre_feats],
    }

    # #################### [ TASKS ] ####################
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    EMB_DIM = 30

    primary_task = FactorizationTaskWrapper(
        loss_fn='bpr',
        sample_method='uniform_verified',
        interactions=xn_train,
        group_features=primary_group_features,
        embedding_map_kwargs={
            'embedding_dim': EMB_DIM,
        },
        batch_size=128,
        optimizer=opt,
        name='primary',
    )

    primary_validator = Validator(
        {'interactions_val': xn_test},
        parent_task_wrapper=primary_task,
        **{
            'limit_items': -1,
            'n_users_eval': 200,
            'include_cold': False,
            'cold_only': False
        },
        name='userXmovie',
    )

    return primary_task, primary_validator


def test_movielens_basic(data):
    """
    Make sure the minimal movielens example works
    Warning: takes > 10s on modern machines
    """
    with tempfile.TemporaryDirectory() as log_dir:

        primary_task, primary_validator = data

        model = TophatModel(tasks=[primary_task])

        summary_cb = cbks.Summary(log_dir=log_dir)

        val_cb = cbks.Scorer(primary_validator,
                             summary_writer=summary_cb.summary_writer,
                             freq=5,)
        callbacks = [
            summary_cb,
            val_cb,
        ]

        model.fit(10, callbacks=callbacks, verbose=False)

    # Training loss should be decreasing between epochs
    assert (np.diff(model.loss_hists[0].epoch_losses) < 0).all()

    # Validation scores at last epoch should not be garbage
    assert val_cb.score_df.iloc[-1]['auc'] > 0.84
    assert val_cb.score_df.iloc[-1]['mapk'] > 0.04

    # Predictions API should work and make sense
    user0_pos_scores = model.predict(0, [170, 201])  # pos from test set
    user0_neg_scores = model.predict(0, [1, 2, 3, 4])  # unseen (full set)
    assert user0_pos_scores.mean() > user0_neg_scores.mean()

