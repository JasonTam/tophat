"""
Implements a generator for basic uniform random sampling of negative items
"""
import numpy as np
from tiefrex.constants import *


def batcher(iterable, n=1):
    """ Generates fixed-size chunks (will not yield last chunk if too small)
    """
    l = len(iterable)
    for ii in range(0, l // n * n, n):
        yield iterable[ii:min(ii + n, l)]


def feed_dicter(
        shuffle_inds, batch_size,
        interactions_df,
        user_col, item_col,
        user_feats_codes_df, item_feats_codes_df,
        item_ids,
        input_pair_d,
):
    while True:
        np.random.shuffle(shuffle_inds)
        inds_batcher = batcher(shuffle_inds, n=batch_size)
        for inds_batch in inds_batcher:
            interactions_batch = interactions_df.iloc[inds_batch]
            user_ids = interactions_batch[user_col].values
            pos_item_ids = interactions_batch[item_col].values
            # Uniform (does not even check that the neg is not a pos)
            neg_item_ids = np.random.choice(item_ids, batch_size)

            user_feed_d = user_feats_codes_df.loc[user_ids].to_dict(
                orient='list')
            pos_item_feed_d = item_feats_codes_df.loc[pos_item_ids].to_dict(
                orient='list')
            neg_item_feed_d = item_feats_codes_df.loc[neg_item_ids].to_dict(
                orient='list')

            feed_pair_dict = {
                **{input_pair_d[f'{USER_VAR_TAG}.{feat_name}']: data_in
                   for feat_name, data_in in user_feed_d.items()},
                **{input_pair_d[f'{POS_VAR_TAG}.{feat_name}']: data_in
                   for feat_name, data_in in pos_item_feed_d.items()},
                **{input_pair_d[f'{NEG_VAR_TAG}.{feat_name}']: data_in
                   for feat_name, data_in in neg_item_feed_d.items()},
            }

            yield feed_pair_dict
