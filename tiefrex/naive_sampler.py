"""
Implements a generator for basic uniform random sampling of negative items
"""
import numpy as np
from collections import defaultdict
from tiefrex.constants import *


def batcher(iterable, n=1):
    """ Generates fixed-size chunks (will not yield last chunk if too small)
    """
    l = len(iterable)
    for ii in range(0, l // n * n, n):
        yield iterable[ii:min(ii + n, l)]


def feat_lookup(ids, feat_code_d):
    ret = defaultdict(list)
    for id_i in ids:
        for k, v in feat_code_d[id_i].items():
            ret[k].append(v)
    return ret


def feed_dicter(
        shuffle_inds, batch_size,
        interactions_df,
        user_col, item_col,
        user_feats_codes_df, item_feats_codes_df,
        item_ids,
        input_pair_d,
):
    # Upfront Conversion
    user_feats_codes_d = user_feats_codes_df.to_dict(orient='index')
    item_feats_codes_d = item_feats_codes_df.to_dict(orient='index')

    while True:
        np.random.shuffle(shuffle_inds)
        inds_batcher = batcher(shuffle_inds, n=batch_size)
        for inds_batch in inds_batcher:
            interactions_batch = interactions_df.iloc[inds_batch]
            user_ids = interactions_batch[user_col].values
            pos_item_ids = interactions_batch[item_col].values
            # Uniform (does not even check that the neg is not a pos)
            neg_item_ids = np.random.choice(item_ids, batch_size)

            user_feed_d = feat_lookup(user_ids, user_feats_codes_d)
            pos_item_feed_d = feat_lookup(pos_item_ids, item_feats_codes_d)
            neg_item_feed_d = feat_lookup(neg_item_ids, item_feats_codes_d)

            feed_pair_dict = {
                **{input_pair_d[f'{USER_VAR_TAG}.{feat_name}']: data_in
                   for feat_name, data_in in user_feed_d.items()},
                **{input_pair_d[f'{POS_VAR_TAG}.{feat_name}']: data_in
                   for feat_name, data_in in pos_item_feed_d.items()},
                **{input_pair_d[f'{NEG_VAR_TAG}.{feat_name}']: data_in
                   for feat_name, data_in in neg_item_feed_d.items()},
            }
            yield feed_pair_dict

