"""
Implements a generator for basic uniform random sampling of negative items
"""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from typing import Dict, Iterable, Sized, Sequence, Optional

from tophat.constants import *
from tophat.data import TrainDataLoader
from tophat.sampling import uniform, adaptive
from tophat.utils.sparse_utils import get_row_nz
from tophat.utils.pseudo_rating import calc_pseudo_ratings


def batcher(seq: Sized, n: int=1):
    """Generates fixed-size chunks (will not yield last chunk if too small)
    
    Args:
        seq: Sequence to batchify
        n: Batch size

    Yields:
        Batch sequence

    """
    l = len(seq)
    for ii in range(0, l // n * n, n):
        yield seq[ii:min(ii + n, l)]


def feed_via_pair(input_pair_d: Dict[str, tf.Tensor],
                  user_feed_d: Dict[str, Iterable],
                  pos_item_feed_d: Dict[str, Iterable],
                  neg_item_feed_d: Dict[str, Iterable],
                  context_feed_d: Dict[str, Iterable],
                  ):
    feed_pair_dict = {
        **{input_pair_d[f'{USER_VAR_TAG}.{feat_name}']: data_in
           for feat_name, data_in in user_feed_d.items()},
        **{input_pair_d[f'{POS_VAR_TAG}.{feat_name}']: data_in
           for feat_name, data_in in pos_item_feed_d.items()},
        **{input_pair_d[f'{NEG_VAR_TAG}.{feat_name}']: data_in
           for feat_name, data_in in neg_item_feed_d.items()},
        **{input_pair_d[f'{CONTEXT_VAR_TAG}.{feat_name}']: data_in
           for feat_name, data_in in context_feed_d.items()},
    }
    return feed_pair_dict


def feed_via_inds(inds_batch: Sequence[int],
                  cols: Sequence[str],
                  codes_arr: np.array,
                  num_arr: np.array,
                  num_key: Optional[str],
                  ):
    """Gets the appropriate data slices for a batch of inds
    (typically, this is the data to be placed in values of a feed dictionary)

    Args:
        inds_batch: Indices of the batch to slice on
        cols: Names of columns to consider
            Ex. user_cols, or item_cols
        codes_arr: Encoded categorical features array
            [n_total_samples x n_categorical_features]
        num_arr: Numerical features array
            [n_total_samples x n_numerical_features]
        num_key: Numerical features key (for book-keeping)
            Ex. 'item_num_feats'

    Returns:
        Dictionary of batch data

    """

    if codes_arr is None:
        return {}
    d = dict(zip(cols, codes_arr[inds_batch, :].T))
    if num_arr is not None and num_key is not None:
        d[num_key] = num_arr[inds_batch, :]
    return d


class PairSampler(object):
    """Convenience class for generating (pos, neg) interaction pairs using
    negative sampling

    Args:
        
        input_pair_d: Dictionary of placeholders keyed by name
        batch_size: Batch size
        shuffle: If `True`, batches will be sampled from a shuffled index
        n_epochs: Number of epochs until `StopIteration`
        uniform_users: If `True` sample by user
            rather than by positive interaction
            (optimize all users equally rather than weighing more active users)
        method: Negative sampling method
        model: Optional model for adaptive sampling
        seed: Seed for random state
    """
    def __init__(self,
                 interactions_df: pd.DataFrame,
                 cols_d: Dict[str, str],
                 cats_d: Dict[str, List],
                 feat_codes_df_d: Dict[str, pd.DataFrame],
                 feats_d_d: Dict[str, Dict[FType, pd.DataFrame]],
                 input_pair_d: Dict[str, tf.Tensor],
                 batch_size: int=1024,
                 shuffle: bool=True,
                 n_epochs: int=-1,
                 uniform_users: bool=False,
                 method: str='uniform',
                 model=None,
                 sess: tf.Session=None,
                 seed: int=0,
                 ):

        self.seed = seed
        np.random.seed(self.seed)

        user_col = cols_d['user']
        item_col = cols_d['item']

        # Index alignment
        user_feats_codes_df = feat_codes_df_d['user'].loc[cats_d[user_col]]
        item_feats_codes_df = feat_codes_df_d['item'].loc[cats_d[item_col]]

        # TODO: some switch for context existence
        # context features are already aligned with `interaction_df`
        #   by construction
        context_feat_codes_df = feat_codes_df_d.get('context')

        # Grab underlying numerical feature array(s)
        self.user_num_feats_arr = None
        self.item_num_feats_arr = None
        if feats_d_d and FType.NUM in feats_d_d['user']:
            self.user_num_feats_arr = feats_d_d['user'][FType.NUM].loc[cats_d[
                    user_col]].values
        if feats_d_d and FType.NUM in feats_d_d['item']:
            self.item_num_feats_arr = feats_d_d['item'][FType.NUM].loc[cats_d[
                    item_col]].values
        # TODO: NUM not supported for context right now

        self.method = method
        self.get_negs = {
            'uniform': self.sample_uniform,
            'uniform_verified': self.sample_uniform_verified,
            'uniform_ordinal': self.sample_uniform_ordinal,
            'adaptive': self.sample_adaptive,
            'adaptive_ordinal': self.sample_adaptive_ordinal,
        }[self.method]

        self.n_epochs = n_epochs if n_epochs >= 0 else sys.maxsize
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.uniform_users = uniform_users

        self.input_pair_d = input_pair_d
        self._model = model

        # Upfront processing
        self.n_users = len(interactions_df[user_col].cat.categories)
        self.n_items = len(interactions_df[item_col].cat.categories)
        if self.method in {'uniform_ordinal', 'adaptive'}:
            df = calc_pseudo_ratings(
                interactions_df=interactions_df,
                user_col=user_col,
                item_col=item_col,
                counts_col='counts',
                weight_switch_col='activity',
                sublinear=True,
                reagg_counts=False,
                output_col='pseudo_rating',
            )

            self.xn_coo = sp.coo_matrix(
                (df['pseudo_rating'],
                 (df[user_col].cat.codes,
                  df[item_col].cat.codes)),
                shape=(self.n_users, self.n_items), dtype=np.float32)
        else:
            self.xn_coo = sp.coo_matrix(
                (np.ones(len(interactions_df), dtype=bool),
                 (interactions_df[user_col].cat.codes,
                  interactions_df[item_col].cat.codes)),
                shape=(self.n_users, self.n_items), dtype=bool)

        if self.method in {'uniform_verified',
                           'uniform_ordinal',
                           'adaptive',
                           'adaptive_ordinal',
                           } \
                or self.uniform_users:
            self.xn_csr = self.xn_coo.tocsr()
        else:
            self.xn_csr = None

        if self.uniform_users:
            # index for each user
            self.shuffle_inds = np.arange(self.n_users)
        else:
            # index for each pos interaction
            self.shuffle_inds = np.arange(len(self.xn_coo.data))

        self.user_feats_codes_arr = user_feats_codes_df.values
        self.item_feats_codes_arr = item_feats_codes_df.values
        self.context_feats_codes_arr = context_feat_codes_df.values \
            if context_feat_codes_df is not None else None
        self.user_cols = user_feats_codes_df.columns
        self.item_cols = item_feats_codes_df.columns
        self.context_cols = context_feat_codes_df.columns \
            if context_feat_codes_df is not None else []

        if 'adaptive' in self.method:
            self.max_sampled = 32  # for WARP

            # Re-usable -- just get it once
            # Flexible batch_size for negative sampling
            #   which will pass in batch_size * max_sampled records
            self.fwd_dict = self._model.get_fwd_dict(batch_size=None)
            self.fwd_op = self._model.forward(self.fwd_dict)

        self.sess = sess

    @classmethod
    def from_data_loader(cls,
                         train_data_loader: TrainDataLoader,
                         input_pair_d: Dict[str, tf.Tensor],
                         batch_size: int=1024,
                         shuffle: bool=True,
                         n_epochs: int=-1,
                         uniform_users: bool=False,
                         method: str='uniform',
                         model=None,
                         seed: int=0,
                         ):
        return cls(
            interactions_df=train_data_loader.interactions_df,
            cols_d={
                'user': train_data_loader.user_col,
                'item': train_data_loader.item_col,
            },
            cats_d=train_data_loader.cats_d,
            feat_codes_df_d={
                'user': train_data_loader.user_feats_codes_df,
                'item': train_data_loader.item_feats_codes_df,
                'context': train_data_loader.context_feats_codes_df,
            },
            feats_d_d={
                'user': train_data_loader.user_feats_d,
                'item': train_data_loader.item_feats_d,
            },
            input_pair_d=input_pair_d,
            batch_size=batch_size,
            shuffle=shuffle,
            n_epochs=n_epochs,
            uniform_users=uniform_users,
            method=method,
            model=model,
            seed=seed,
        )

    def __iter__(self):
        if self.uniform_users:
            return self.iter_by_user()
        else:
            return self.iter_by_xn()

    def sample_uniform(self, **_):
        """See tophat.uniform.sample_uniform"""
        return uniform.sample_uniform(self.n_items, self.batch_size)

    def sample_uniform_verified(self,
                                user_inds_batch: Sequence[int],
                                **_):
        """See tophat.uniform.sample_uniform_verified"""
        return uniform.sample_uniform_verified(self.n_items,
                                               self.xn_csr,
                                               user_inds_batch)

    def sample_uniform_ordinal(self,
                               user_inds_batch: Sequence[int],
                               pos_item_inds_batch: Sequence[int],
                               **_):
        return uniform.sample_uniform_ordinal(
            self.n_items,
            self.xn_csr,
            user_inds_batch,
            pos_item_inds_batch,
        )

    def sample_adaptive(self,
                        user_inds_batch: Sequence[int],
                        pos_item_inds_batch: Sequence[int],
                        use_first_violation: bool=False,
                        ):
        """See tophat.adaptive.sample_adaptive"""
        return adaptive.sample_adaptive(self.n_items,
                                        self.max_sampled,
                                        self.score_via_inds_fn,
                                        user_inds_batch,
                                        pos_item_inds_batch,
                                        use_first_violation,
                                        None,
                                        )

    def sample_adaptive_ordinal(self,
                                user_inds_batch: Sequence[int],
                                pos_item_inds_batch: Sequence[int],
                                use_first_violation: bool=False,
                                ):
        """See tophat.adaptive.sample_adaptive"""
        return adaptive.sample_adaptive(self.n_items,
                                        self.max_sampled,
                                        self.score_via_inds_fn,
                                        user_inds_batch,
                                        pos_item_inds_batch,
                                        use_first_violation,
                                        self.xn_csr,
                                        )

    def score_via_dict_fn(self, fwd_dict):
        return self.sess.run(self.fwd_op, feed_dict=fwd_dict)

    def score_via_inds_fn(self,
                          user_inds,
                          item_inds,
                          ):
        fwd_dict = self.fwd_dicter_via_inds(user_inds,
                                            item_inds,
                                            self.fwd_dict)
        return self.score_via_dict_fn(fwd_dict)

    def user_feed_via_inds(self, user_inds_batch):
        return feed_via_inds(user_inds_batch,
                             self.user_cols,
                             self.user_feats_codes_arr,
                             self.user_num_feats_arr,
                             num_key='user_num_feats',
                             )

    def item_feed_via_inds(self, item_inds_batch):
        return feed_via_inds(item_inds_batch,
                             self.item_cols,
                             self.item_feats_codes_arr,
                             self.item_num_feats_arr,
                             num_key='item_num_feats',
                             )

    def context_feed_via_inds(self, inds_batch):
        return feed_via_inds(inds_batch,
                             self.context_cols,
                             self.context_feats_codes_arr,
                             num_arr=None,
                             num_key=None,
                             )

    def iter_by_xn(self):
        # The feed dict generator itself
        # Note: can implement __next__ as well
        #   if we want book-keeping state info to be kept
        for i in range(self.n_epochs):
            if self.shuffle:
                np.random.shuffle(self.shuffle_inds)
            # TODO: problem if less inds than batch_size
            inds_batcher = batcher(self.shuffle_inds, n=self.batch_size)
            for inds_batch in inds_batcher:
                user_inds_batch = self.xn_coo.row[inds_batch]
                pos_item_inds_batch = self.xn_coo.col[inds_batch]
                neg_item_inds_batch = self.get_negs(
                    user_inds_batch=user_inds_batch,
                    pos_item_inds_batch=pos_item_inds_batch,)

                user_feed_d = self.user_feed_via_inds(user_inds_batch)
                pos_item_feed_d = self.item_feed_via_inds(pos_item_inds_batch)
                neg_item_feed_d = self.item_feed_via_inds(neg_item_inds_batch)

                context_feed_d = self.context_feed_via_inds(inds_batch)

                feed_pair_dict = feed_via_pair(
                    self.input_pair_d,
                    user_feed_d,
                    pos_item_feed_d, neg_item_feed_d,
                    context_feed_d
                )
                yield feed_pair_dict

    def iter_by_user(self):
        # The feed dict generator itself
        # Note: can implement __next__ as well
        #   if we want book-keeping state info to be kept
        for i in range(self.n_epochs):
            if self.shuffle:
                np.random.shuffle(self.shuffle_inds)
            # TODO: problem if less inds than batch_size
            inds_batcher = batcher(self.shuffle_inds, n=self.batch_size)
            for inds_batch in inds_batcher:
                # TODO: WIP>>>
                user_inds_batch = inds_batch
                pos_l = []

                for user_ind in user_inds_batch:
                    user_pos_item_inds = get_row_nz(self.xn_csr, user_ind)
                    # `random.choice` slow
                    user_pos_item = user_pos_item_inds[np.random.randint(
                        len(user_pos_item_inds))]
                    pos_l.append(user_pos_item)
                # Select random known pos for user
                # pos_item_inds_batch = self.xn_coo.col[inds_batch]
                pos_item_inds_batch = np.array(pos_l)
                neg_item_inds_batch = self.get_negs(
                    user_inds_batch=user_inds_batch,
                    pos_item_inds_batch=pos_item_inds_batch,)

                user_feed_d = self.user_feed_via_inds(user_inds_batch)
                pos_item_feed_d = self.item_feed_via_inds(pos_item_inds_batch)
                neg_item_feed_d = self.item_feed_via_inds(neg_item_inds_batch)

                # TODO: fix context feed
                feed_pair_dict = feed_via_pair(
                    self.input_pair_d,
                    user_feed_d,
                    pos_item_feed_d, neg_item_feed_d,
                    context_feed_d={},
                )
                yield feed_pair_dict

    def fwd_dicter_via_inds(self,
                            user_inds: Union[int, Sequence[int]],
                            item_inds: Sequence[int],
                            fwd_d: Dict[int, tf.Tensor],
                            ):
        """Forward inference dictionary via indices of users and items

        Args:
            user_inds: Can be a single user ind or an iterable of user inds
                If a single user ind is provided, it will be repeated 
                for each item in `item_inds`
            item_inds: Item indices
            fwd_d: Dictionary of placeholders for forward inference

        Returns:
            Feed forward dictionary

        """

        if not hasattr(user_inds, '__iter__'):
            user_inds = [user_inds] * len(item_inds)
        user_feed_d = dict(zip(self.user_cols,
                               self.user_feats_codes_arr[user_inds, :].T))
        item_feed_d = dict(zip(self.item_cols,
                               self.item_feats_codes_arr[item_inds, :].T))
        feed_fwd_dict = {
            **{fwd_d[f'{feat_name}']: data_in
               for feat_name, data_in in user_feed_d.items()},
            **{fwd_d[f'{feat_name}']: data_in
               for feat_name, data_in in item_feed_d.items()},
        }
        return feed_fwd_dict
