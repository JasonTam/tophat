""" Tests batch negative sampling stuff

TODO:
    - adaptive (might need to mock a lot of stuff)
    - ordinal

"""
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from tophat.sampling.pair_sampler import PairSampler
from tophat.constants import FGroup
from pandas.api.types import CategoricalDtype

N_BATCHES_TEST = 5


@pytest.fixture(params=['uniform', 'uniform_verified'])
def data(request):
    cats_d = {
        'user_id': ['u1', 'u2'],
        'item_id': ['i1', 'i2', 'i3', 'i4', 'i5'],
    }

    interactions_df = pd.DataFrame([
        ['u1', 'i1', 'a1', 1],
        ['u1', 'i1', 'a1', 2],
        ['u1', 'i2', 'a1', 1],
        ['u1', 'i3', 'a1', 1],
        ['u1', 'i3', 'a2', 1],
        ['u2', 'i2', 'a2', 1],
    ], columns=['user_id', 'item_id', 'activity', 'count']
    )
    interactions_df['user_id'] = interactions_df['user_id'].astype(
        CategoricalDtype(cats_d['user_id']))
    interactions_df['item_id'] = interactions_df['item_id'].astype(
        CategoricalDtype(cats_d['item_id']))

    cols_d = {
        FGroup.USER: 'user_id',
        FGroup.ITEM: 'item_id',
        'activity': 'activity',
        'count': 'count',
    }

    # For the sake of testing convenience,
    # the first code is going to be the index
    feat_codes_df_d = {
        FGroup.USER: pd.DataFrame([
            [0, 1, 5],
            [1, 2, 5], ],
            columns=['user_feat0_code', 'user_feat1_code', 'user_feat2_code'],
            index=pd.Index(cats_d['user_id'], name='user_id')),
        FGroup.ITEM: pd.DataFrame([
            [0, 11, 50],
            [1, 12, 50],
            [2, 13, 50],
            [3, 14, 50],
            [4, 15, 50], ],
            columns=['item_feat0_code', 'item_feat1_code', 'item_feat2_code'],
            index=pd.Index(cats_d['item_id'], name='item_id')),
    }

    # Only needed for NUM
    feats_d_d = {
        FGroup.USER: {},
        FGroup.ITEM: {},
    }

    batch_size = 2

    input_pair_d = {
        k: tf.placeholder('int32', (batch_size,)) for k in
        ['user.user_id',
         'pos.genre_id',
         'pos.item_id',
         'neg.genre_id',
         'neg.item_id', ]
    }

    tf.placeholder('int32', (batch_size,))

    sampler = PairSampler(
        interactions_df,
        cols_d,
        cats_d,
        feat_codes_df_d,
        feats_d_d,
        input_pair_d,
        batch_size,
        method=request.param,
    )

    return sampler, cats_d, interactions_df, feat_codes_df_d


def get_inds(sampled_batch, i):
    user_ind = sampled_batch['user.user_feat0_code'][i]
    pos_item_ind = sampled_batch['pos.item_feat0_code'][i]
    neg_item_ind = sampled_batch['neg.item_feat0_code'].T[i]
    return user_ind, pos_item_ind, neg_item_ind


def test_feat_alignment(data):
    """
    Feature codes should have stayed aligned
    """
    sampler, cats_d, interactions_df, feat_codes_df_d = data
    gen = sampler.__iter__()
    for batch_i in range(N_BATCHES_TEST):
        sampled_batch = next(gen)

        for i in range(sampler.batch_size):
            user_ind, pos_item_ind, neg_item_ind = get_inds(sampled_batch, i)

            np.testing.assert_array_equal(
                feat_codes_df_d[FGroup.USER].iloc[user_ind],
                [sampled_batch[f'user.{col}'][i]
                 for col in feat_codes_df_d[FGroup.USER].columns]
            )

            np.testing.assert_array_equal(
                feat_codes_df_d[FGroup.ITEM].iloc[pos_item_ind],
                [sampled_batch[f'pos.{col}'][i]
                 for col in feat_codes_df_d[FGroup.ITEM].columns]
            )


def test_pos(data):
    """
    User should have interaction history with pos item
    """
    sampler, cats_d, interactions_df, feat_codes_df_d = data
    gen = sampler.__iter__()
    for batch_i in range(N_BATCHES_TEST):
        sampled_batch = next(gen)
        for i in range(sampler.batch_size):
            user_ind, pos_item_ind, neg_item_ind = get_inds(sampled_batch, i)
            user_id = cats_d['user_id'][user_ind]
            pos_item_id = cats_d['item_id'][pos_item_ind]
            user_xn = interactions_df.loc[interactions_df['user_id'] == user_id]

            assert pos_item_id in user_xn['item_id'].values


def test_neg(data):
    """
    User should have no interaction history with neg item
    """
    sampler, cats_d, interactions_df, feat_codes_df_d = data
    if sampler.method in {'uniform', 'adaptive'}:
        pytest.skip()

    gen = sampler.__iter__()
    for batch_i in range(N_BATCHES_TEST):
        sampled_batch = next(gen)
        for i in range(sampler.batch_size):
            user_ind, pos_item_ind, neg_item_ind = get_inds(sampled_batch, i)
            user_id = cats_d['user_id'][user_ind]
            neg_item_id = np.array(cats_d['item_id'])[neg_item_ind]
            user_xn = interactions_df.loc[interactions_df['user_id'] == user_id]

            assert not set(neg_item_id).intersection(user_xn['item_id'].values)
