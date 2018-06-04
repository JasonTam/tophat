import pytest
import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
import tophat.callbacks as cbks
from pathlib import Path
from tophat.data import FeatureSource, InteractionsSource
from tophat.constants import FType, FGroup
from tophat.tasks.wrapper import FactorizationTaskWrapper
from tophat.core import TophatModel
from tophat.evaluation import Validator
from tophat.utils.io import load_vocab

from tophat.datasets.movielens import fetch_movielens


@pytest.fixture
def data():
    tf.reset_default_graph()
    # Get movielens data via lightfm
    movielens = fetch_movielens(
        indicator_features=False,
        genre_features=True,
        # This will include more ratings than the checkpoint that we will load
        # (checkpoint was trained on `min_rating=5.0`
        # so this test needs to handle newly seen users & items as well
        min_rating=4.0,
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

    # #################### [ LOAD PRETRAINED EMBS ] ####################
    ckpt_path = str(Path(__file__).parent / 'data/movielens/model.ckpt')

    vocab_d = {}
    # Columns to load embeddings for
    transfer_cols = ['user_id', 'item_id']
    for feat_name in transfer_cols:
        for scope in ['embeddings', 'biases']:
            tensor_name = f'{scope}/{feat_name}'
            vocab_file = str(
                Path(__file__).parent / f'data/movielens/{feat_name}.vocab')
            vocab_d[tensor_name] = vocab_file

    existing_cats = load_vocab('tests/data/movielens/')

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
            'init_emb_via_vocab': vocab_d,
            'path_checkpoint': ckpt_path,
        },
        existing_cats=existing_cats,
        add_new_cats=True,
        batch_size=128,
        optimizer=opt,
        name='primary',
    )

    primary_validator = Validator(
        xn_test,
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


def test_movielens_loaded(data):
    """
    Make sure we can load from checkpoint on evolving vocabulary
    Warning: takes > 5s on modern machines
    """
    with tempfile.TemporaryDirectory() as log_dir:

        primary_task, primary_validator = data

        summary_cb = cbks.Summary(log_dir=log_dir)

        model = TophatModel(tasks=[primary_task])
        model.sess_init()
        score_d = primary_validator.run_val(
            model.sess, summary_cb.summary_writer, step=0, macro=False)

    # Scores should be better than random since we loaded lots of info
    # Original validation was ~0.84, we expect something worse since we
    # added some cold-start and did not train any additional epochs
    assert score_d['auc'] > 0.75

