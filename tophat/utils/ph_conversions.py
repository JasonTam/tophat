import tensorflow as tf
from typing import Iterable
from tophat.constants import *


def fwd_dict_via_cats(cat_keys: Iterable[str],
                      batch_size: int=None) -> Dict[str, tf.Tensor]:
    """Creates placeholders for forward inference
    
    Args:
        cat_keys: Feature names
        batch_size: Optional batch size

    Returns:
        Dictionary of placeholders
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
    """Creates placeholders for paired loss
    
    Args:
        user_cat_cols: Name of user categorical feature columns
        item_cat_cols: Name of item categorical feature columns
        batch_size: Optional batch size

    Returns:
        Dictionary of placeholders
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
                      dtype: Union[str, type],
                      batch_size: int=None,
                      input_size: int=1,
                      tag: str=None,
                      ) -> Dict[str, tf.Tensor]:
    """Creates placeholders based on desired features
    
    Args:
        feat_names: Name of features to consider
        dtype: Data type of the placeholder
        batch_size: Optional batch size
        input_size: Size of non-batch dimension
            (typically used for numerical features, else, 1)
        tag: Prefix tag for the placeholder
            (ex. "user", "neg")

    Returns:
        Dictionary of placeholders
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
    """Creates placeholders based on feature type metadata 
    
    Args:
        ftypemeta: Feature type metadata
        batch_size: Optional batch size
        tag: Prefix tag for the placeholder
            (ex. "user", "neg")

    Returns:
        Dictionary of placeholders
    """
    cat_d = ph_dict_via_feats(ftypemeta[FType.CAT],
                              dtype=tf.int32,
                              batch_size=batch_size,
                              tag=tag)
    if FType.NUM in ftypemeta:
        num_ph_l = [
            ph_dict_via_feats([feat_name],
                              dtype=tf.float32,
                              batch_size=batch_size,
                              input_size=feat_size,
                              tag=tag)
            for feat_name, feat_size in ftypemeta[FType.NUM]]
        num_d = {k: v for d in num_ph_l for k, v in d.items()}
    else:
        num_d = {}
    return {**cat_d, **num_d}


def pair_dict_via_ftypemeta(
        user_ftypemeta: FtypeMeta,
        item_ftypemeta: FtypeMeta,
        context_ftypemeta: FtypeMeta,
        batch_size: int=None) -> Dict[str, tf.Tensor]:
    """Creates placeholders based on feature type metadata
    for a pair of interactions
    
    Args:
        user_ftypemeta: User feature type metadata
        item_ftypemeta: Item feature type metadata
        context_ftypemeta: Context feature type metadata
        batch_size: Optional batch size

    Returns:
        Dictionary of placeholders
    """
    input_pair_d = {
        **ph_via_ftypemeta(user_ftypemeta, batch_size, tag=USER_VAR_TAG),
        **ph_via_ftypemeta(item_ftypemeta, batch_size, tag=POS_VAR_TAG),
        **ph_via_ftypemeta(item_ftypemeta, batch_size, tag=NEG_VAR_TAG),
        **ph_via_ftypemeta(context_ftypemeta, batch_size, tag=CONTEXT_VAR_TAG),
    }
    return input_pair_d


fwd_dict_via_ftypemeta = ph_via_ftypemeta
