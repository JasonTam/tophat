import tensorflow as tf
import pandas as pd
from typing import Any, Dict, Sequence, Optional, Union, Generator, Tuple

from tophat.constants import FGroup


def items_pred_dicter(user_id: Any, item_ids: Sequence[Any],
                      cat_codes_dfs: Dict[FGroup, pd.DataFrame],
                      num_feats_dfs: Dict[FGroup, pd.DataFrame],
                      context_ind: Optional[int] = None,
                      input_fwd_d: Optional[Dict[str, tf.Tensor]] = None,
                      ) -> Dict[Union[str, tf.Tensor], Any]:
    # todo: a little redundancy with the dicters in tophat.core
    """Creates feeds for forward prediction for a single user
    Note: This does not batch within the list of items passed in
    thus, it could be a problem with huge number of items

    Args:
        user_id: The particular user we are predicting for
        item_ids: Item ids to predict over
        cat_codes_dfs: Encoded category features
        num_feats_dfs: Numerical features
        context_ind: The particular context we are predicting under
            (index of interaction df that describes the context)
        input_fwd_d: Optional dictionary of feed-forward placeholders
            If provided, will change the keys of the return to
            placeholder tensors.

    Returns:
        Feed dictionary to score all items for a given user under a context

    """

    n_items = len(item_ids)

    ids_by_group = {
        FGroup.USER: [user_id],
        FGroup.ITEM: item_ids,
    }

    feed_d = {}
    for fgroup in [FGroup.USER, FGroup.ITEM]:
        feed_d[fgroup] = cat_codes_dfs[fgroup] \
            .loc[ids_by_group[fgroup]].to_dict(orient='list')

        # Add numerical feature if present
        if num_feats_dfs[fgroup] is not None:
            feed_d[fgroup].update(
                {f'{fgroup}_num_feats':
                     num_feats_dfs[fgroup].loc[ids_by_group[fgroup]].values})

    # TODO: care with loc since it's a subset -- maybe just stick to iloc
    if (FGroup.CONTEXT in cat_codes_dfs.keys()) and \
            (cat_codes_dfs[FGroup.CONTEXT] is not None):
        feed_d[FGroup.CONTEXT] = cat_codes_dfs[FGroup.CONTEXT] \
            .iloc[[context_ind]].to_dict(orient='list')
    else:
        feed_d[FGroup.CONTEXT] = {}

    feed_fwd_dict = {
        **{f'{feat_name}': data_in * n_items
           for feat_name, data_in in feed_d[FGroup.USER].items()},
        **{f'{feat_name}': data_in
           for feat_name, data_in in feed_d[FGroup.ITEM].items()},
        **{f'{feat_name}': data_in * n_items
           for feat_name, data_in in feed_d[FGroup.CONTEXT].items()},
    }

    if input_fwd_d is not None:
        feed_fwd_dict = {input_fwd_d[k]: v for k, v in feed_fwd_dict.items()}

    return feed_fwd_dict


def items_pred_dicter_gen(
        user_ids: Sequence[Any],
        item_ids: Sequence[Any],
        cat_codes_dfs: Dict[FGroup, pd.DataFrame],
        num_feats_dfs: Dict[FGroup, pd.DataFrame],
        input_fwd_d: Optional[Dict[str, tf.Tensor]] = None,
) -> Generator[Tuple[int, Dict[Union[str, tf.Tensor], Any]], None, None]:
    """Generates feeds for forward prediction for many users
        each batch will be all items for a single user

    Args:
        user_ids: User ids to predict over
        item_ids: Item ids to predict over
        cat_codes_dfs: Encoded category features
        num_feats_dfs: Numerical features
        input_fwd_d: Dictionary of feed-forward placeholders

    Yields:
        Tuple of user id, feed forward dictionary for that user
    """

    for user_id in user_ids:
        yield user_id, items_pred_dicter(user_id, item_ids,
                                         cat_codes_dfs,
                                         num_feats_dfs,
                                         context_ind=None,
                                         input_fwd_d=input_fwd_d,
                                         )


def items_pred_dicter_gen_context(
        interaction_df: pd.DataFrame,
        item_ids: Sequence[Any],
        cat_codes_dfs: Dict[FGroup, pd.DataFrame],
        num_feats_dfs: Dict[FGroup, pd.DataFrame],
        context_inds: Sequence[int],
        input_fwd_d: Dict[str, tf.Tensor],
):
    """Generates feeds for forward prediction for many users-contexts
    each batch will be all items for a single user-context

    Args:
        interaction_df: Interactions
        item_ids: Item ids to predict over
        cat_codes_dfs: Encoded category features
        num_feats_dfs: Numerical features
        input_fwd_d: Dictionary of feed-forward placeholders
        context_inds: Context indices

    Yields:
        Tuple of context ind, feed forward dictionary

    """

    def get_user_id(context_ind):
        return interaction_df['ops_user_id'].iloc[context_ind]

    for context_ind in context_inds:
        user_id = get_user_id(context_ind)
        yield context_ind, items_pred_dicter(
            user_id, item_ids,
            cat_codes_dfs,
            num_feats_dfs,
            context_ind=context_ind,
            input_fwd_d=input_fwd_d,
        )
