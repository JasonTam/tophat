import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from tophat.core import pair_dict_via_cols
from tophat.sampling import pair_sampler


class TestPP(unittest.TestCase):
    def setUp(self):
        self.user_col = 'user_id'
        self.item_col = 'item_id'

        self.batch_size = 2

        self.interactions_df = pd.DataFrame([
            {self.user_col: 'u1', self.item_col: 'i1'},
            {self.user_col: 'u2', self.item_col: 'i2'},
            {self.user_col: 'u3', self.item_col: 'i3'},
            {self.user_col: 'u2', self.item_col: 'i3'},
        ])

        self.user_feats_df = pd.DataFrame([
            {self.user_col: 'u1', 'gender': 'm'},
            {self.user_col: 'u2', 'gender': 'f'},
            {self.user_col: 'u3', 'gender': 'm'},
        ]).set_index(self.user_col, drop=False)

        self.item_feats_df = pd.DataFrame([
            {self.item_col: 'i1', 'brand_id': 'b1'},
            {self.item_col: 'i2', 'brand_id': 'b1'},
            {self.item_col: 'i3', 'brand_id': 'b2'},
        ]).set_index(self.item_col, drop=False)

        self.user_feat_cols = self.user_feats_df.columns
        self.item_feat_cols = self.item_feats_df.columns

        self.cats_d = {}
        self.process_cats()

        self.input_pair_d = {
            k: tf.placeholder(dtype='int32', shape=self.batch_size, name=f'{k}_input')
            for k in [
                'user.user_id',
                'user.gender',
                'pos.item_id',
                'pos.brand_id',
                'neg.item_id',
                'neg.brand_id',
            ]}

    def process_cats(self):
        """ Most basic category cast"""
        for col in [self.user_col, self.item_col]:
            self.interactions_df[col] = self.interactions_df[col].astype('category')
            self.cats_d[col] = self.interactions_df[col].cat.categories.tolist()

        for col in self.user_feats_df.columns:
            if col in self.cats_d:
                self.user_feats_df[col] = self.user_feats_df[col].astype(
                    'category', categories=self.cats_d[col])
            else:
                self.user_feats_df[col] = self.user_feats_df[col].astype('category')
                self.cats_d[col] = self.user_feats_df[col].cat.categories.tolist()

        for col in self.item_feats_df.columns:
            if col in self.cats_d:
                self.item_feats_df[col] = self.item_feats_df[col].astype(
                    'category', categories=self.cats_d[col])
            else:
                self.item_feats_df[col] = self.item_feats_df[col].astype('category')
                self.cats_d[col] = self.item_feats_df[col].cat.categories.tolist()

    def test_input_pair_d(self):
        input_pair_d = pair_dict_via_cols(
            self.user_feat_cols, self.item_feat_cols, self.batch_size)

        # Number of placeholders should bet he number of user features + 2* item features (pos and neg)
        self.assertEqual(
            len({placeholder.name for placeholder in input_pair_d.values()}),
            len(self.user_feat_cols) + 2*len(self.item_feat_cols)
        )

        # Each placeholder should be of shape [batch_size]
        self.assertEqual(
            [placeholder.get_shape().as_list() for placeholder in input_pair_d.values()],
            [[self.batch_size]] * (len(self.user_feat_cols) + 2*len(self.item_feat_cols))
        )

        # Compare to expected explicitly (above assertions to make sure our hard-code isn't messed up)
        for input_name, input_place in input_pair_d.items():
            self.assertEqual(input_pair_d[input_name].get_shape(), input_place.get_shape())
            self.assertEqual(input_pair_d[input_name].dtype, input_place.dtype)

    def test_feed_gen(self):
        # Convert everything to categorical codes
        user_feats_codes_df = self.user_feats_df.copy()
        for col in user_feats_codes_df.columns:
            user_feats_codes_df[col] = user_feats_codes_df[col].cat.codes
        item_feats_codes_df = self.item_feats_df.copy()
        for col in item_feats_codes_df.columns:
            item_feats_codes_df[col] = item_feats_codes_df[col].cat.codes

        item_ids_all = self.cats_d[self.item_col]

        shuffle_inds = np.arange(len(self.interactions_df))
        feed_dict_gen = pair_sampler.feed_dicter(
            shuffle_inds, self.batch_size,
            self.interactions_df,
            self.user_col, self.item_col,
            user_feats_codes_df, item_feats_codes_df,
            item_ids_all, self.input_pair_d)

        # Make sure things are OK when the gen has to repeat itself
        # ~ n_steps*batch_size/len(interactions_df) = 10./4. n_epochs
        for n_steps in range(5):
            batch_in = next(feed_dict_gen)

            user_inds = batch_in[self.input_pair_d['user.user_id']]
            user_ids = [self.cats_d[self.user_col][ii] for ii in user_inds]

            # Make sure user feature codes from the generator are aligned properly
            self.assertEqual(
                self.user_feats_df.loc[user_ids].gender.cat.codes.tolist(),
                batch_in[self.input_pair_d['user.gender']]
            )

            # Check items
            pos_item_inds = batch_in[self.input_pair_d['pos.item_id']]
            pos_item_ids = [self.cats_d[self.item_col][ii] for ii in pos_item_inds]
            neg_item_inds = batch_in[self.input_pair_d['neg.item_id']]
            neg_item_ids = [self.cats_d[self.item_col][ii] for ii in neg_item_inds]

            # Make sure item feature codes from the generator are aligned properly
            self.assertEqual(
                self.item_feats_df.loc[pos_item_ids].brand_id.cat.codes.tolist(),
                batch_in[self.input_pair_d['pos.brand_id']]
            )
            self.assertEqual(
                self.item_feats_df.loc[neg_item_ids].brand_id.cat.codes.tolist(),
                batch_in[self.input_pair_d['neg.brand_id']]
            )

            # Make sure the pos item is actually in the interactions
            for user_id, item_id in zip(user_ids, pos_item_ids):
                self.assertTrue(
                    item_id in
                    self.interactions_df.loc[
                        self.interactions_df[self.user_col] == user_id][self.item_col
                    ].values
                )

if __name__ == '__main__':
    unittest.main()

