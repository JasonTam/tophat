"""
Implements a generator for basic uniform random sampling of negative items
"""
import sys
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tiefrex.constants import *
from tiefrex.utils_sparse import get_row_nz
from typing import Dict, Iterable, Sized


def batcher(iterable: Sized, n=1):
    """ Generates fixed-size chunks (will not yield last chunk if too small)
    """
    l = len(iterable)
    for ii in range(0, l // n * n, n):
        yield iterable[ii:min(ii + n, l)]


def feed_via_pair(input_pair_d: Dict[str, tf.Tensor],
                  user_feed_d: Dict[str, Iterable],
                  pos_item_feed_d: Dict[str, Iterable],
                  neg_item_feed_d: Dict[str, Iterable]):
    feed_pair_dict = {
        **{input_pair_d[f'{USER_VAR_TAG}.{feat_name}']: data_in
           for feat_name, data_in in user_feed_d.items()},
        **{input_pair_d[f'{POS_VAR_TAG}.{feat_name}']: data_in
           for feat_name, data_in in pos_item_feed_d.items()},
        **{input_pair_d[f'{NEG_VAR_TAG}.{feat_name}']: data_in
           for feat_name, data_in in neg_item_feed_d.items()},
    }
    return feed_pair_dict


class PairSampler(object):
    def __init__(self,
                 train_data_loader,
                 input_pair_d: Dict[str, tf.Tensor],
                 batch_size: int=1024,
                 shuffle: bool=True,
                 n_epochs: int=-1,
                 method: str='uniform',
                 net=None,
                 ):
        """
        :param train_data_loader: tiefrex.core.TrainDataLoader
        :param input_pair_d: dictionary of placeholders keyed by name
        :param batch_size: batch size
        :param shuffle: if `True`, batches will be sampled from a shuffled index
        :param n_epochs: number of epochs until `StopIteration`
        :param method: negative sampling method
        :param net: network object that implements a forward method for adaptive sampling
        """
        interactions_df = train_data_loader.interactions_df
        user_col = train_data_loader.user_col
        item_col = train_data_loader.item_col
        user_feats_codes_df = train_data_loader.user_feats_codes_df.loc[
            train_data_loader.cats_d[train_data_loader.user_col]]
        item_feats_codes_df = train_data_loader.item_feats_codes_df.loc[
            train_data_loader.cats_d[train_data_loader.item_col]]
        self.method = method
        self.get_negs = {
            'uniform': self.sample_uniform,
            'uniform_verified': self.sample_uniform_verified,
            'adaptive': self.sample_adaptive,
        }[self.method]

        self.n_epochs = n_epochs if n_epochs >= 0 else sys.maxsize
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.input_pair_d = input_pair_d
        self._net = net

        # Upfront processing
        self.n_users = len(interactions_df[user_col].cat.categories)
        self.n_items = len(interactions_df[item_col].cat.categories)
        self.xn_coo = sp.coo_matrix(
            (np.ones(len(interactions_df), dtype=bool),
             (interactions_df[user_col].cat.codes,
              interactions_df[item_col].cat.codes)),
            shape=(self.n_users, self.n_items), dtype=bool)

        if self.method in {'uniform_verified', 'adaptive'}:
            self.xn_csr = self.xn_coo.tocsr()
        else:
            self.xn_csr = None

        self.shuffle_inds = np.arange(len(self.xn_coo.data))

        self.user_feats_codes_arr = user_feats_codes_df.values
        self.item_feats_codes_arr = item_feats_codes_df.values
        self.user_cols = user_feats_codes_df.columns
        self.item_cols = item_feats_codes_df.columns

        # Some more crap for the more complex strats
        if self.method == 'adaptive':
            self.sess = tf.Session()
            self.max_sampled = 128  # for WARP
            self.pos_fwd_d = self._net.get_fwd_dict(batch_size=self.batch_size)
            self.pos_fwd_op = self._net.forward(self.pos_fwd_d)
            self.neg_fwd_d = self._net.get_fwd_dict(batch_size=self.max_sampled)
            self.neg_fwd_op = self._net.forward(self.neg_fwd_d)

            self.sess.run(tf.global_variables_initializer())

    def sample_uniform(self, **_):
        return np.random.randint(self.n_items, size=self.batch_size)

    def sample_uniform_verified(self, user_inds_batch, **_):
        """ Ensures that the neg samples are not known positives
        Consider just using `sample_uniform` as this can be ~20x slower
        """
        neg_item_inds_batch = []
        for user_ind in user_inds_batch:
            user_pos_item_inds = get_row_nz(self.xn_csr, user_ind)
            neg_item_ind = np.random.randint(self.n_items)
            while neg_item_ind in user_pos_item_inds:
                neg_item_ind = np.random.randint(self.n_items)
            neg_item_inds_batch.append(neg_item_ind)
        return neg_item_inds_batch

    def sample_adaptive(self, user_inds_batch, pos_item_inds_batch):
        """
        Uses the forward prediction of `self._net` to sample the first violating negative
        Note: If WARP, we'll also return the number of samples we passed through
         TODO: need to handle this return signature somehow
        """
        neg_item_inds_batch = []
        pos_fwd_dict = self.fwd_dicter_via_inds(
            user_inds_batch, pos_item_inds_batch, self.pos_fwd_d)
        pos_scores = self.sess.run(self.pos_fwd_op, feed_dict=pos_fwd_dict)
        for user_ind, pos_score in zip(user_inds_batch, pos_scores):
            user_pos_item_inds = get_row_nz(self.xn_csr, user_ind)
            # Sample all candidates right away
            neg_item_inds = np.random.randint(self.n_items, size=self.max_sampled)
            verification = np.array(
                [ind not in user_pos_item_inds for ind in neg_item_inds], dtype=bool)
            neg_cand_fwd_dict = self.fwd_dicter_via_inds(user_ind, neg_item_inds, self.neg_fwd_d)
            neg_cand_scores = self.sess.run(self.neg_fwd_op, feed_dict=neg_cand_fwd_dict)

            violations = (neg_cand_scores > pos_score - 1) & verification
            # Get index of the first violation
            first_violator_ind = np.argmax(violations)
            if ~violations[first_violator_ind]:
                # There were no violations
                first_violator_ind = self.max_sampled - 1
            neg_item_inds_batch.append(neg_item_inds[first_violator_ind])
            # TODO: here is also where we would return the number of samples to it took to reach a violation
            #   = first_violator_ind - (~verification[:first_violator_ind]).sum()
        return neg_item_inds_batch

    def sample_most_violating_candidate(self, user_inds_batch, n_candidates=10):
        """
        Uses the forward prediction of `self._net` to sample the most violating candidate from
        a set of random candidates
        """
        raise NotImplementedError

    def __iter__(self):
        # The feed dict generator itself
        # Note: can implement __next__ as well if we want book-keeping state info to be kept
        for i in range(self.n_epochs):
            if self.shuffle:
                np.random.shuffle(self.shuffle_inds)
            inds_batcher = batcher(self.shuffle_inds, n=self.batch_size)
            for inds_batch in inds_batcher:
                user_inds_batch = self.xn_coo.row[inds_batch]
                pos_item_inds_batch = self.xn_coo.col[inds_batch]
                neg_item_inds_batch = self.get_negs(
                    user_inds_batch=user_inds_batch,
                    pos_item_inds_batch=pos_item_inds_batch,)

                user_feed_d = dict(
                    zip(self.user_cols, self.user_feats_codes_arr[user_inds_batch, :].T))
                pos_item_feed_d = dict(
                    zip(self.item_cols, self.item_feats_codes_arr[pos_item_inds_batch, :].T))
                neg_item_feed_d = dict(
                    zip(self.item_cols, self.item_feats_codes_arr[neg_item_inds_batch, :].T))

                feed_pair_dict = feed_via_pair(
                    self.input_pair_d, user_feed_d, pos_item_feed_d, neg_item_feed_d)
                yield feed_pair_dict

    def fwd_dicter_via_inds(self, user_inds, item_inds, fwd_d):
        """
        :param user_inds: Can be a single user ind or an iterable of user inds
            If a single user ind is provided, it will be repeated for each item in `item_inds`
        :param item_inds: Iterable of item inds
        :param fwd_d: forward feed dict template
        """
        if not hasattr(user_inds, '__iter__'):
            user_inds = [user_inds] * len(item_inds)
        user_feed_d = dict(zip(self.user_cols, self.user_feats_codes_arr[user_inds, :].T))
        item_feed_d = dict(zip(self.item_cols, self.item_feats_codes_arr[item_inds, :].T))
        feed_fwd_dict = {
            **{fwd_d[f'{feat_name}']: data_in
               for feat_name, data_in in user_feed_d.items()},
            **{fwd_d[f'{feat_name}']: data_in
               for feat_name, data_in in item_feed_d.items()},
        }
        return feed_fwd_dict
