"""
Implements a generator for basic uniform random sampling of negative items
"""
import sys
import numpy as np
from collections import defaultdict
from tiefrex.constants import *
from typing import Dict, Iterable, Sized, Generator
from tensorflow import Tensor


def batcher(iterable: Sized, n=1):
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


def feed_via_pair(input_pair_d: Dict[str, Tensor],
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


def feed_dicter(
        shuffle_inds: Sized, batch_size: int,
        interactions_df,
        user_col: str, item_col: str,
        user_feats_codes_df, item_feats_codes_df,
        item_ids: Iterable,
        input_pair_d: Dict[str, Tensor],
        shuffle: bool=True,
        n_epochs: int=-1,
) -> Generator[Dict[Tensor, Iterable], None, None]:
    if n_epochs < 0:
        n_epochs = sys.maxsize

    # Upfront Conversion
    user_feats_codes_d = user_feats_codes_df.to_dict(orient='index')
    item_feats_codes_d = item_feats_codes_df.to_dict(orient='index')

    for i in range(n_epochs):
        if shuffle:
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

            feed_pair_dict = feed_via_pair(
                input_pair_d, user_feed_d, pos_item_feed_d, neg_item_feed_d)
            yield feed_pair_dict
