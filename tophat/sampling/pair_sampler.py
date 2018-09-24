"""
Implements a generator for basic uniform random sampling of negative items
"""
import sys

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.sparse as sp
import tensorflow as tf
from collections import ChainMap
from typing import Iterable, Sized, Sequence, Optional

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


def feed_via_pair(
        user_feed_d: Dict[str, Iterable],
        pos_item_feed_d: Dict[str, Iterable],
        neg_item_feed_d: Dict[str, Iterable],
        context_feed_d: Dict[str, Iterable],
        misc_feed_d: Optional[Dict[str, Iterable]] = None,
        input_pair_d: Optional[Dict[str, tf.Tensor]] = None,
        ) -> Dict[str, np.array]:

    feed_pair_dict = dict(ChainMap(
        *[{f'{tag}.{feat_name}': data_in
           for feat_name, data_in in feed_d.items()}
          for tag, feed_d in [
              (USER_VAR_TAG, user_feed_d),
              (POS_VAR_TAG, pos_item_feed_d),
              (NEG_VAR_TAG, neg_item_feed_d),
              (CONTEXT_VAR_TAG, context_feed_d),
              (MISC_TAG, misc_feed_d),
          ] if feed_d is not None]
    ))

    if input_pair_d is not None:
        feed_pair_dict = {
            input_pair_d[k]: v for k, v in feed_pair_dict.items()}

    return feed_pair_dict


def feed_via_inds(inds_batch: Sequence[int],
                  cols: Sequence[str],
                  codes_arr: Optional[np.array],
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
        use_ds_iter: If `True`, use tf.data.Dataset iterator API, else
            use generator of placeholder dictionaries for feed_dict API
        seed: Seed for random state
        non_negs_df: Additional interactions that are safeguarded from being
            sampled as negatives. But they will not be chosen as positives.
        n_neg: number of negatives to sample per positive

    Terminology:

        - *positives*: the interactions to sample from
        - *non-negatives*: interactions that are safeguarded against being
          sampled as negatives. But not necessarily sampled as positives.
          _Usually_ though, they will be the same.

    """
    def __init__(self,
                 interactions_df: pd.DataFrame,
                 cols_d: Dict[Union[FGroup, str], str],
                 cats_d: Dict[str, List],
                 feat_codes_df_d: Dict[FGroup, pd.DataFrame],
                 feats_d_d: Dict[FGroup, Dict[FType, pd.DataFrame]],
                 input_pair_d: Dict[str, tf.Tensor],
                 batch_size: int = 1024,
                 shuffle: bool = True,
                 n_epochs: int = -1,
                 uniform_users: bool = False,
                 method: str = 'uniform',
                 model=None,
                 sess: tf.Session = None,
                 use_ds_iter: bool = True,
                 seed: int = 0,
                 non_negs_df: Optional[pd.DataFrame] = None,
                 n_neg: int = 1,
                 ):

        self.rand = np.random.RandomState(seed)

        user_col = cols_d[FGroup.USER]
        item_col = cols_d[FGroup.ITEM]
        activity_col = cols_d['activity']
        count_col = cols_d['count']

        # Index alignment
        feats_codes_dfs = {
            # TODO: fishy... this breaks sometimes if changed to `reindex`
            fg: feat_codes_df_d[fg].loc[cats_d[cols_d[fg]]]
            for fg in [FGroup.USER, FGroup.ITEM]
        }

        # Grab underlying numerical feature array(s)
        self.user_num_feats_arr = None
        self.item_num_feats_arr = None
        if feats_d_d and FType.NUM in feats_d_d[FGroup.USER]:
            self.user_num_feats_arr = feats_d_d[FGroup.USER][FType.NUM]\
                .loc[cats_d[user_col]].values
        if feats_d_d and FType.NUM in feats_d_d[FGroup.ITEM]:
            self.item_num_feats_arr = feats_d_d[FGroup.ITEM][FType.NUM]\
                .loc[cats_d[item_col]].values
        # TODO: NUM not supported for context right now

        self.method = method
        self.get_negs = {
            'uniform': self.sample_uniform,
            'uniform_verified': self.sample_uniform_verified,
            'uniform_ordinal': self.sample_uniform_ordinal,
            'adaptive': self.sample_adaptive,
            'adaptive_ordinal': self.sample_adaptive_ordinal,
            'adaptive_warp': self.sample_adaptive_warp,
        }[self.method]

        self.n_epochs = n_epochs if n_epochs >= 0 else sys.maxsize
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.uniform_users = uniform_users

        self.input_pair_d = input_pair_d
        self.use_ds_iter = use_ds_iter
        self.input_pair_d_usage = None if self.use_ds_iter \
            else self.input_pair_d
        self._model = model

        # Upfront processing
        self.n_users = len(interactions_df[user_col].cat.categories)
        self.n_items = len(interactions_df[item_col].cat.categories)

        self.pos_xn_coo = sp.coo_matrix(
            (np.ones(len(interactions_df), dtype=bool),
             (interactions_df[user_col].cat.codes,
              interactions_df[item_col].cat.codes)),
            shape=(self.n_users, self.n_items), dtype=bool)

        if non_negs_df is not None:
            # Additional non-negs passed in
            # should match interaction cats
            for col in [user_col, item_col]:
                non_negs_df[col] = non_negs_df[col].astype(CategoricalDtype(
                    categories=interactions_df[col].cat.categories))
            non_negs_df.dropna(inplace=True)

            non_negs_df = pd.concat([non_negs_df, interactions_df], axis=0)
        else:
            non_negs_df = interactions_df

        # Methods that require non-neg verification
        if self.method in {'uniform_verified',
                           'uniform_ordinal',
                           'adaptive',
                           'adaptive_ordinal',
                           'adaptive_warp',
                           }:

            # Pseudo-ratings for non-neg
            if ('ordinal' in self.method and
                    activity_col in interactions_df.columns):
                non_negs_pr_df = calc_pseudo_ratings(
                    interactions_df=non_negs_df,
                    user_col=user_col,
                    item_col=item_col,
                    counts_col=count_col,
                    weight_switch_col=activity_col,
                    sublinear=True,
                    reagg_counts=False,
                    output_col='pseudo_rating',
                )

                self.non_neg_xn_csr = sp.csr_matrix(
                    (non_negs_pr_df['pseudo_rating'],
                     (non_negs_pr_df[user_col].cat.codes,
                      non_negs_pr_df[item_col].cat.codes)),
                    shape=(self.n_users, self.n_items), dtype=np.float32)
            else:
                self.non_neg_xn_csr = sp.csr_matrix(
                    (np.ones(len(non_negs_df), dtype=bool),
                     (non_negs_df[user_col].cat.codes,
                      non_negs_df[item_col].cat.codes)),
                    shape=(self.n_users, self.n_items), dtype=bool)
        else:
            self.non_neg_xn_csr = None

        if self.uniform_users:
            # index for each user
            self.shuffle_inds = np.arange(self.n_users)
        else:
            # index for each pos interaction
            self.shuffle_inds = np.arange(len(self.pos_xn_coo.data))

        self.feats_codes_arrs = {
            fg: df.values if hasattr(df, 'values') else None
            for fg, df in feats_codes_dfs.items()
        }
        self.code_df_cols = {
            fg: df.columns if hasattr(df, 'columns') else None
            for fg, df in feats_codes_dfs.items()
        }
        self.user_cols = feats_codes_dfs[FGroup.USER].columns
        self.item_cols = feats_codes_dfs[FGroup.ITEM].columns
        self.context_cols = feats_codes_dfs[FGroup.CONTEXT].columns \
            if (
            FGroup.CONTEXT in feats_codes_dfs and
            feats_codes_dfs[FGroup.CONTEXT] is not None) else []

        if 'adaptive' in self.method:
            self.max_sampled = 32  # for WARP

            # Re-usable -- just get it once
            # Flexible batch_size for negative sampling
            #   which will pass in batch_size * max_sampled records
            self.fwd_dict = self._model.get_fwd_dict(batch_size=None)
            self.fwd_op = self._model.forward(self.fwd_dict)

        self.n_neg = n_neg

        self.sess = sess

    @classmethod
    def from_data_loader(cls,
                         train_data_loader: TrainDataLoader,
                         input_pair_d: Dict[str, tf.Tensor],
                         batch_size: int = 1024,
                         shuffle: bool = True,
                         n_epochs: int = -1,
                         uniform_users: bool = False,
                         method: str = 'uniform',
                         model=None,
                         use_ds_iter: bool = True,
                         seed: int = 0,
                         non_negs_df: Optional[pd.DataFrame] = None,
                         ):
        return cls(
            interactions_df=train_data_loader.interactions_df,
            cols_d=train_data_loader.cols,
            cats_d=train_data_loader.cats_d,
            feat_codes_df_d=train_data_loader.feats_codes_df,
            feats_d_d=train_data_loader.feats_by_group,
            input_pair_d=input_pair_d,
            batch_size=batch_size,
            shuffle=shuffle,
            n_epochs=n_epochs,
            uniform_users=uniform_users,
            method=method,
            model=model,
            use_ds_iter=use_ds_iter,
            seed=seed,
            non_negs_df=non_negs_df,
        )

    def __iter__(self):
        if self.uniform_users:
            return self.iter_by_user()
        else:
            return self.iter_by_xn()

    def sample_uniform(self, **_):
        """See :func:`tophat.sampling.uniform.sample_uniform`"""
        return uniform.sample_uniform(self.n_items, self.batch_size, self.n_neg,)

    def sample_uniform_verified(self,
                                user_inds_batch: Sequence[int],
                                **_):
        """See :func:`tophat.sampling.uniform.sample_uniform_verified`"""
        return uniform.sample_uniform_verified(self.n_items,
                                               self.non_neg_xn_csr,
                                               user_inds_batch,
                                               self.n_neg,
                                               )

    def sample_uniform_ordinal(self,
                               user_inds_batch: Sequence[int],
                               pos_item_inds_batch: Sequence[int],
                               **_):
        """See :func:`tophat.sampling.uniform.sample_uniform_ordinal`"""
        return uniform.sample_uniform_ordinal(
            self.n_items,
            self.non_neg_xn_csr,
            user_inds_batch,
            pos_item_inds_batch,
            self.n_neg,
        )

    def sample_adaptive(self,
                        user_inds_batch: Sequence[int],
                        pos_item_inds_batch: Sequence[int],
                        use_first_violation: bool = False,
                        ):
        """See :func:`tophat.sampling.adaptive.sample_adaptive`"""
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
                                use_first_violation: bool = False,
                                ):
        """See :func:`tophat.sampling.adaptive.sample_adaptive`"""
        return adaptive.sample_adaptive(self.n_items,
                                        self.max_sampled,
                                        self.score_via_inds_fn,
                                        user_inds_batch,
                                        pos_item_inds_batch,
                                        use_first_violation,
                                        self.non_neg_xn_csr,
                                        )

    def sample_adaptive_warp(self,
                             user_inds_batch: Sequence[int],
                             pos_item_inds_batch: Sequence[int],
                             use_first_violation: bool = True,
                             return_n_samp: bool = True,
                             ):
        """See :func:`tophat.sampling.adaptive.sample_adaptive`"""
        return adaptive.sample_adaptive(self.n_items,
                                        self.max_sampled,
                                        self.score_via_inds_fn,
                                        user_inds_batch,
                                        pos_item_inds_batch,
                                        use_first_violation,
                                        self.non_neg_xn_csr,
                                        return_n_samp,
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
                             self.code_df_cols[FGroup.USER],
                             self.feats_codes_arrs[FGroup.USER],
                             self.user_num_feats_arr,
                             num_key='user_num_feats',
                             )

    def item_feed_via_inds(self, item_inds_batch):
        return feed_via_inds(item_inds_batch,
                             self.code_df_cols[FGroup.ITEM],
                             self.feats_codes_arrs[FGroup.ITEM],
                             self.item_num_feats_arr,
                             num_key='item_num_feats',
                             )

    def context_feed_via_inds(self, inds_batch):
        return feed_via_inds(inds_batch,
                             self.code_df_cols.get(FGroup.CONTEXT, None),
                             self.feats_codes_arrs.get(FGroup.CONTEXT, None),
                             num_arr=None,
                             num_key=None,
                             )

    def iter_by_xn(self):
        # The feed dict generator itself
        # Note: can implement __next__ as well
        #   if we want book-keeping state info to be kept
        for i in range(self.n_epochs):
            if self.shuffle:
                self.rand.shuffle(self.shuffle_inds)
            # TODO: problem if less inds than batch_size
            inds_batcher = batcher(self.shuffle_inds, n=self.batch_size)
            for inds_batch in inds_batcher:
                user_inds_batch = self.pos_xn_coo.row[inds_batch]
                pos_item_inds_batch = self.pos_xn_coo.col[inds_batch]
                if self.method == 'adaptive_warp':
                    neg_item_inds_batch, first_violator_inds = self.get_negs(
                        user_inds_batch=user_inds_batch,
                        pos_item_inds_batch=pos_item_inds_batch,)
                    misc_feed_d = {'first_violator_inds': first_violator_inds}
                else:
                    neg_item_inds_batch = self.get_negs(
                        user_inds_batch=user_inds_batch,
                        pos_item_inds_batch=pos_item_inds_batch,
                    )
                    misc_feed_d = None

                user_feed_d = self.user_feed_via_inds(user_inds_batch)
                pos_item_feed_d = self.item_feed_via_inds(pos_item_inds_batch)
                neg_item_feed_d = self.item_feed_via_inds(neg_item_inds_batch)

                context_feed_d = self.context_feed_via_inds(inds_batch)

                feed_pair_dict = feed_via_pair(
                    user_feed_d,
                    pos_item_feed_d, neg_item_feed_d,
                    context_feed_d,
                    misc_feed_d=misc_feed_d,
                    input_pair_d=self.input_pair_d_usage,
                )
                yield feed_pair_dict

    def iter_by_user(self):
        # The feed dict generator itself
        # Note: can implement __next__ as well
        #   if we want book-keeping state info to be kept

        pos_xn_csr = self.pos_xn_coo.tocsr()
        for i in range(self.n_epochs):
            if self.shuffle:
                self.rand.shuffle(self.shuffle_inds)
            # TODO: problem if less inds than batch_size
            inds_batcher = batcher(self.shuffle_inds, n=self.batch_size)
            for inds_batch in inds_batcher:
                # TODO: WIP>>>
                user_inds_batch = inds_batch
                pos_l = []

                for user_ind in user_inds_batch:
                    user_pos_item_inds = get_row_nz(pos_xn_csr, user_ind)
                    # `random.choice` slow
                    user_pos_item = user_pos_item_inds[self.rand.randint(
                        len(user_pos_item_inds))]
                    pos_l.append(user_pos_item)
                # Select random known pos for user
                # pos_item_inds_batch = self.xn_coo.col[inds_batch]
                pos_item_inds_batch = np.array(pos_l)
                if self.method == 'adaptive_warp':
                    neg_item_inds_batch, first_violator_inds = self.get_negs(
                        user_inds_batch=user_inds_batch,
                        pos_item_inds_batch=pos_item_inds_batch,)
                    misc_feed_d = {'first_violator_inds': first_violator_inds}
                else:
                    neg_item_inds_batch = self.get_negs(
                        user_inds_batch=user_inds_batch,
                        pos_item_inds_batch=pos_item_inds_batch,)
                    misc_feed_d = None

                user_feed_d = self.user_feed_via_inds(user_inds_batch)
                pos_item_feed_d = self.item_feed_via_inds(pos_item_inds_batch)
                neg_item_feed_d = self.item_feed_via_inds(neg_item_inds_batch)

                # TODO: fix context feed
                feed_pair_dict = feed_via_pair(
                    user_feed_d,
                    pos_item_feed_d, neg_item_feed_d,
                    context_feed_d={},
                    misc_feed_d=misc_feed_d,
                    input_pair_d=self.input_pair_d_usage,
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

        inds_d = {
            FGroup.USER: user_inds,
            FGroup.ITEM: item_inds,
        }

        feed_fwd_dict = {}
        for fg in [FGroup.USER, FGroup.ITEM]:
            inds = inds_d[fg]
            feed_d = dict(zip(self.code_df_cols[fg],
                              self.feats_codes_arrs[fg][inds, :].T))

            feed_fwd_dict.update(
                {fwd_d[f'{feat_name}']: data_in
                 for feat_name, data_in in feed_d.items()}
            )

        return feed_fwd_dict
