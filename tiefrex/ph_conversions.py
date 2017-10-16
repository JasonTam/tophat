import tensorflow as tf
from typing import Iterable
from tiefrex.constants import *


def fwd_dict_via_cats(cat_keys: Iterable[str],
                      batch_size: int=None) -> Dict[str, tf.Tensor]:
    """ Creates placeholders for forward inference
    :param cat_keys: feature names
    :param batch_size: 
    :return: dictionary of placeholders
    """
    input_forward_d = {
        feat_name: tf.placeholder(tf.int32, shape=[batch_size],
                                  name=f'{feat_name}_input')
        for feat_name in cat_keys
    }
    return input_forward_d


def pair_dict_via_cols(user_cat_cols: Iterable[str],
                       item_cat_cols: Iterable[str],
                       batch_size: int=None) -> Dict[str, tf.Tensor]:
    """ Creates placeholders for paired loss
    :param user_cat_cols: 
    :param item_cat_cols: 
    :param batch_size: 
    :return: 
    """
    input_pair_d = {
        **{f'{USER_VAR_TAG}.{feat_name}': tf.placeholder(
            tf.int32, shape=[batch_size],
            name=f'{USER_VAR_TAG}.{feat_name}_input')
            for feat_name in user_cat_cols},
        **{f'{POS_VAR_TAG}.{feat_name}': tf.placeholder(
            tf.int32, shape=[batch_size],
            name=f'{POS_VAR_TAG}.{feat_name}_input')
            for feat_name in item_cat_cols},
        **{f'{NEG_VAR_TAG}.{feat_name}': tf.placeholder(
            tf.int32, shape=[batch_size],
            name=f'{NEG_VAR_TAG}.{feat_name}_input')
            for feat_name in item_cat_cols}
    }
    return input_pair_d


def ph_dict_via_feats(feat_names: List[str],
                      dtype,
                      batch_size: int=None,
                      input_size=1,
                      tag: str=None,
                      ) -> Dict[str, tf.Tensor]:
    """ Creates a dictionary of placeholders keyed by feature name 
    """
    if not feat_names:
        return {}

    if input_size > 1:
        ph_shape = [batch_size, input_size]
    else:
        ph_shape = [batch_size]

    if tag:
        def name(s): return f'{tag}.{s}_input'

        def key(s): return f'{tag}.{s}'
    else:
        def name(s): return f'{s}_input'

        def key(s): return f'{s}'

    return {key(feat_name): tf.placeholder(
        dtype, shape=ph_shape, name=name(feat_name))
        for feat_name in feat_names}


def ph_via_ftypemeta(ftypemeta: FtypeMeta,
                     batch_size: int=None,
                     tag: str=None) -> Dict[str, tf.Tensor]:
    cat_d = ph_dict_via_feats(ftypemeta[FType.CAT],
                              dtype=tf.int32,
                              batch_size=batch_size,
                              tag=tag)
    if FType.NUM in ftypemeta:
        num_ph_l = [
            ph_dict_via_feats([tup[0]],
                              dtype=tf.float32,
                              batch_size=batch_size,
                              tag=tag, input_size=tup[1])
            for tup in ftypemeta[FType.NUM]]
        num_d = {k: v for d in num_ph_l for k, v in d.items()}
    else:
        num_d = {}
    return {**cat_d, **num_d}


def pair_dict_via_ftypemeta(
        user_ftypemeta: FtypeMeta,
        item_ftypemeta: FtypeMeta,
        context_ftypemeta: FtypeMeta,
        batch_size: int=None) -> Dict[str, tf.Tensor]:
    input_pair_d = {
        **ph_via_ftypemeta(user_ftypemeta, batch_size, tag=USER_VAR_TAG),
        **ph_via_ftypemeta(item_ftypemeta, batch_size, tag=POS_VAR_TAG),
        **ph_via_ftypemeta(item_ftypemeta, batch_size, tag=NEG_VAR_TAG),
        **ph_via_ftypemeta(context_ftypemeta, batch_size, tag=CONTEXT_VAR_TAG),
    }
    return input_pair_d


def fwd_dict_via_ftypemeta(
        ftypemeta: FtypeMeta, batch_size: int=None) -> Dict[str, tf.Tensor]:
    return ph_via_ftypemeta(ftypemeta, batch_size, tag=None)
