from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.metrics import streaming_mean, \
    streaming_sparse_average_precision_at_k
from tqdm import tqdm
from typing import Dict, Any, Generator, Tuple, Sequence, Optional, Union

from tophat.constants import FType
from tophat.data import load_simple_warm_cats, load_simple, TrainDataLoader
from tophat.utils.pp_utils import append_dt_extracts
from tophat.utils.log import logger
from tophat.core import FactModel


def items_pred_dicter(user_id: Any, item_ids: Sequence[Any],
                      user_cat_codes_df: pd.DataFrame,
                      item_cat_codes_df: pd.DataFrame,
                      user_num_feats_df: pd.DataFrame,
                      item_num_feats_df: pd.DataFrame,
                      context_ind: int=None,
                      context_cat_codes_df: pd.DataFrame=None,
                      # context_num_feats: pd.DataFrame=None,
                      input_fwd_d: Optional[Dict[str, tf.Tensor]]=None,
                      ) -> Dict[Union[str, tf.Tensor], Any]:
    # todo: a little redundancy with the dicters in tophat.core
    """Creates feeds for forward prediction for a single user
    Note: This does not batch within the list of items passed in
    thus, it could be a problem with huge number of items

    Args:
        user_id: The particular user we are predicting for
        item_ids: Item ids to predict over
        user_cat_codes_df: Encoded user category features 
        item_cat_codes_df: Encoded item category features
        user_num_feats_df: User numerical features
        item_num_feats_df: Item numerical features
        context_ind: The particular context we are predicting under
        context_cat_codes_df: Dataframe of encoded context category features
        input_fwd_d: Optional dictionary of feed-forward placeholders
            If provided, will change the keys of the return to 
            placeholder tensors. 

    Returns:
        Feed dictionary to score all items for a given user under a context

    """

    n_items = len(item_ids)

    user_feed_d = user_cat_codes_df.loc[[user_id]].to_dict(orient='list')
    item_feed_d = item_cat_codes_df.loc[item_ids].to_dict(orient='list')

    # TODO: care with loc since it's a subset -- maybe just stick to iloc
    context_feed_d = context_cat_codes_df.iloc[[context_ind]].to_dict(
        orient='list') if context_cat_codes_df is not None else {}

    # Add numerical feature if present
    if user_num_feats_df is not None:
        user_feed_d.update(
            {'user_num_feats': user_num_feats_df.loc[[user_id]].values})
    if item_num_feats_df is not None:
        item_feed_d.update(
            {'item_num_feats': item_num_feats_df.loc[item_ids].values})

    feed_fwd_dict = {
        **{f'{feat_name}': data_in * n_items
           for feat_name, data_in in user_feed_d.items()},
        **{f'{feat_name}': data_in
           for feat_name, data_in in item_feed_d.items()},
        **{f'{feat_name}': data_in * n_items
           for feat_name, data_in in context_feed_d.items()},
    }

    if input_fwd_d is not None:
        feed_fwd_dict = {input_fwd_d[k]: v for k, v in feed_fwd_dict.items()}

    return feed_fwd_dict


def items_pred_dicter_gen(
        user_ids: Sequence[Any],
        item_ids: Sequence[Any],
        user_cat_codes_df: pd.DataFrame,
        item_cat_codes_df: pd.DataFrame,
        user_num_feats_df: pd.DataFrame,
        item_num_feats_df: pd.DataFrame,
        input_fwd_d: Optional[Dict[str, tf.Tensor]],
) -> Generator[Tuple[int, Dict[tf.Tensor, Any]], None, None]:
    """Generates feeds for forward prediction for many users
        each batch will be all items for a single user
    
    Args:
        user_ids: User ids to predict over
        item_ids: Item ids to predict over
        user_cat_codes_df: Encoded user category features 
        item_cat_codes_df: Encoded item category features
        user_num_feats_df: User numerical features
        item_num_feats_df: Item numerical features
        input_fwd_d: Dictionary of feed-forward placeholders

    Yields:
        Tuple of user id, feed forward dictionary for that user 
    """

    for user_id in user_ids:
        yield user_id, items_pred_dicter(user_id, item_ids,
                                         user_cat_codes_df, item_cat_codes_df,
                                         user_num_feats_df, item_num_feats_df,
                                         context_ind=None,
                                         context_cat_codes_df=None,
                                         # context_num_feats=context_num_feats,
                                         input_fwd_d=input_fwd_d,
                                         )


def items_pred_dicter_gen_context(
        interaction_df: pd.DataFrame,
        item_ids: Sequence[Any],
        user_cat_codes_df: pd.DataFrame,
        item_cat_codes_df: pd.DataFrame,
        user_num_feats_df: pd.DataFrame,
        item_num_feats_df: pd.DataFrame,
        context_inds: Sequence[int],
        context_cat_codes_df: pd.DataFrame,
        input_fwd_d: Dict[str, tf.Tensor],
):
    """Generates feeds for forward prediction for many users-contexts
    each batch will be all items for a single user-context

    Args:
        interaction_df: Interactions
        item_ids: Item ids to predict over
        user_cat_codes_df: Encoded user category features 
        item_cat_codes_df: Encoded item category features
        user_num_feats_df: User numerical features
        item_num_feats_df: Item numerical features
        input_fwd_d: Dictionary of feed-forward placeholders
        context_inds: Context indices
        context_cat_codes_df: Encoded context category features

    Yields:
        Tuple of context ind, feed forward dictionary
        
    """

    def get_user_id(context_ind):
        return interaction_df['ops_user_id'].iloc[context_ind]

    for context_ind in context_inds:
        user_id = get_user_id(context_ind)
        yield context_ind, items_pred_dicter(
            user_id, item_ids,
            user_cat_codes_df, item_cat_codes_df,
            user_num_feats_df, item_num_feats_df,
            context_ind=context_ind,
            context_cat_codes_df=context_cat_codes_df,
            input_fwd_d=input_fwd_d,
        )


def make_metrics_ops(fwd_op: tf.Tensor,
                     input_fwd_d: Dict[str, tf.Tensor]):
    """Makes operations for calculating metrics
    
    Args:
        fwd_op: Forward operation of network
        input_fwd_d: Dictionary of feed-forward placeholders
            Optionally, this will also contain target variables

    Returns:
        Tuple of metric operations, reset operations, and feed dictionary
    """

    if (('y_true_ph' in input_fwd_d.keys()) and
            ('y_true_bool_ph' in input_fwd_d.keys())):
        targ_d = {
            'y_true_ph': input_fwd_d['y_true_ph'],
            'y_true_bool_ph': input_fwd_d['y_true_bool_ph'],
        }
    else:
        with tf.name_scope('placeholders_eval'):
            targ_d = {
                'y_true_ph': tf.placeholder('int64'),
                'y_true_bool_ph': tf.placeholder('bool'),
            }

    # Define our metrics: MAP@10 and AUC (hardcoded for now)
    k = 10
    with tf.name_scope('stream_metrics'):
        val_preds = tf.expand_dims(fwd_op(input_fwd_d), 0)
        # NB: THESE STREAMING METRICS DO **MICRO** UPDATES
        # Ex) for AUC, it will effectively concat all predictions and all trues
        #   (over all users)
        # instead of averaging the AUC's over users
        mapk, update_op_mapk = streaming_sparse_average_precision_at_k(
            val_preds, targ_d['y_true_ph'], k=k)
        auc, update_op_auc = tf.metrics.auc(
            targ_d['y_true_bool_ph'], tf.sigmoid(val_preds))
        # Tjur's Pseudo R2 inspired bpr
        pb_t = tf.to_float(targ_d['y_true_bool_ph'],
                           name='true_pos')
        nb_t = tf.to_float(tf.logical_not(targ_d['y_true_bool_ph']),
                           name='true_neg')
        pos_mean = (tf.reduce_sum(tf.multiply(pb_t, val_preds)) /
                    tf.reduce_sum(pb_t))
        neg_mean = (tf.reduce_sum(tf.multiply(nb_t, val_preds)) /
                    tf.reduce_sum(nb_t))
        tjurs_bpr_i = tf.subtract(1., tf.sigmoid(pos_mean - neg_mean),
                                  name='tjurs_bpr_val')
        tjur, update_op_tjur = streaming_mean(tjurs_bpr_i)
        pm, update_op_pm = streaming_mean(pos_mean)
        nm, update_op_nm = streaming_mean(neg_mean)

    stream_vars = [i for i in tf.local_variables()
                   if i.name.split('/')[0] == 'stream_metrics']
    reset_metrics_op = [tf.variables_initializer(stream_vars)]

    metric_ops_d = {
        'mapk': (mapk, update_op_mapk),
        'auc': (auc, update_op_auc),
        'tjurs': (tjur, update_op_tjur),
        'pm': (pm, update_op_pm),
        'nm': (nm, update_op_nm),
    }
    return metric_ops_d, reset_metrics_op, targ_d


def eval_things(sess,
                interactions_df,
                user_col, item_col,
                user_ids_val, item_ids,
                user_cat_codes_df, item_cat_codes_df,
                user_num_feats_df, item_num_feats_df,
                input_fwd_d,
                metric_ops_d, reset_metrics_op, eval_ph_d,
                n_users_eval=-1,
                summary_writer=None, step=None,
                model=None,  # todo
                inds=None,
                context_cat_codes_df=None,
                ):
    """
    
    Args:
        sess: Tensorflow session
        interactions_df: Validation interactions
        user_col: Name of user column
        item_col: Name of item column
        user_ids_val: User ids to evaluate over
        item_ids: Item ids to evaluate over
        user_cat_codes_df: Encoded user category features 
        item_cat_codes_df: Encoded item category features
        user_num_feats_df: User numerical features
        item_num_feats_df: Item numerical features
        input_fwd_d: Dictionary of feed-forward placeholders
        metric_ops_d: Metric operations
        reset_metrics_op: Reset operation for streaming metrics
        eval_ph_d: Placeholders for evaluation
        n_users_eval: Max number of users to evaluate 
            (if evaluation is too slow to consider all users in `user_ids_val`)
        summary_writer: Summary writer object
        step: Global step for summary
        model: 
        inds: 
        context_cat_codes_df: 

    Returns:

    """

    sess.run(tf.local_variables_initializer())
    sess.run(reset_metrics_op)
    # use the same users for every eval step
    pred_feeder_gen = items_pred_dicter_gen(
        user_ids_val, item_ids,
        user_cat_codes_df, item_cat_codes_df,
        user_num_feats_df, item_num_feats_df,
        input_fwd_d)
    if n_users_eval < 0:
        n_users_eval = len(user_ids_val)
    else:
        n_users_eval = min(n_users_eval, len(user_ids_val))

    macro_metrics = defaultdict(lambda: [])
    for ii in tqdm(range(n_users_eval)):
        try:
            user_id, cur_user_fwd_dict = next(pred_feeder_gen)
        except StopIteration:
            break
        # Use re-encoded instead of `item_col`
        y_true = interactions_df.loc[interactions_df[user_col] == user_id]\
            ['item_reenc'].cat.codes.values

        y_true_bool = np.zeros(len(item_ids), dtype=bool)
        y_true_bool[y_true] = True

        # Run updates
        sess.run([tup[1] for tup in metric_ops_d.values()], feed_dict={
            **cur_user_fwd_dict,
            **{eval_ph_d['y_true_ph']: y_true[None, :],
               eval_ph_d['y_true_bool_ph']: y_true_bool[None, :],
               }})
        # Run and store aggregation
        metric_vals = sess.run([tup[0] for tup in metric_ops_d.values()])
        for m, v in zip(metric_ops_d.keys(), metric_vals):
            macro_metrics[m].append(v)

        sess.run(reset_metrics_op)

    # # NOTE: This is for micro aggregation (also remove the reset above)
    # for m, m_tup in metric_ops_d.items():
    #     metric_score = sess.run(m_tup[0])
    #     metric_val_summary = tf.Summary(value=[
    #         tf.Summary.Value(tag=f'{m}_val',
    #                          simple_value=metric_score)])
    #     logger.info(f'(val){m} = {metric_score}')
    #     if summary_writer is not None:
    #         summary_writer.add_summary(metric_val_summary, step)

    ret_d = {}
    for m, vals in macro_metrics.items():
        metric_score = np.mean(vals)
        metric_score_std = np.std(vals)
        metric_val_summary = tf.Summary(value=[
            tf.Summary.Value(tag=f'{m}_val',
                             simple_value=metric_score)])
        logger.info(f'(val){m} = {metric_score} +/- {metric_score_std}')
        if summary_writer is not None:
            summary_writer.add_summary(metric_val_summary, step)

        ret_d[m] = metric_score

    return ret_d


def eval_things_context(
        sess,
        interactions_df,
        user_col, item_col,
        user_ids_val, item_ids,
        user_cat_codes_df, item_cat_codes_df,
        user_num_feats_df, item_num_feats_df,
        input_fwd_d,
        metric_ops_d, reset_metrics_op, eval_ph_d,
        n_xn_eval=-1,
        summary_writer=None, step=None,
        model=None,  # todo
        context_inds=None,
        context_cat_codes_df=None,
        ):
    # TODO: SEE ABOVE.... THIS IS JUST A COPY PASTA + TEMP MODS

    sess.run(tf.local_variables_initializer())
    sess.run(reset_metrics_op)
    # use the same interactions for every eval step
    pred_feeder_gen = items_pred_dicter_gen_context(
        interactions_df, item_ids,
        user_cat_codes_df, item_cat_codes_df,
        user_num_feats_df, item_num_feats_df,
        input_fwd_d,
        context_inds=context_inds,
        context_cat_codes_df=context_cat_codes_df,
    )
    if n_xn_eval < 0:
        n_xn_eval = len(context_inds)
    else:
        n_xn_eval = min(n_xn_eval, len(context_inds))

    macro_metrics = defaultdict(lambda: [])
    for ii in tqdm(range(n_xn_eval)):
        try:
            context_ind, cur_xn_fwd_dict = next(pred_feeder_gen)
        except StopIteration:
            break
        # y_true = interactions_df.loc[interactions_df[user_col]
        #                              == user_id][item_col].cat.codes.values
        y_true = interactions_df.iloc[[context_ind]]['item_reenc'].cat.codes.values

        y_true_bool = np.zeros(len(item_ids), dtype=bool)
        y_true_bool[y_true] = True
        sess.run([tup[1] for tup in metric_ops_d.values()], feed_dict={
            **cur_xn_fwd_dict,
            **{eval_ph_d['y_true_ph']: y_true[None, :],
               eval_ph_d['y_true_bool_ph']: y_true_bool[None, :],
               }})
        for m, m_tup in metric_ops_d.items():
            macro_metrics[m].append(sess.run(m_tup[0]))
        sess.run(reset_metrics_op)

    # # NOTE: This is for micro aggregation (also remove the reset above)
    # for m, m_tup in metric_ops_d.items():
    #     metric_score = sess.run(m_tup[0])
    #     metric_val_summary = tf.Summary(value=[
    #         tf.Summary.Value(tag=f'{m}_val',
    #                          simple_value=metric_score)])
    #     logger.info(f'(val){m} = {metric_score}')
    #     if summary_writer is not None:
    #         summary_writer.add_summary(metric_val_summary, step)

    for m, vals in macro_metrics.items():
        metric_score = np.mean(vals)
        metric_score_std = np.std(vals)
        metric_val_summary = tf.Summary(value=[
            tf.Summary.Value(tag=f'{m}_val',
                             simple_value=metric_score)])
        logger.info(f'(val){m} = {metric_score} +/- {metric_score_std}')
        if summary_writer is not None:
            summary_writer.add_summary(metric_val_summary, step)

    return macro_metrics


class Validator(object):
    """Convenience validation object with various book-keeping
    
    Args:
        config: 
        train_data_loader: Training data object
        model_ref: reference (training) model used to determine input structure
        limit_items: Limits the number of items in catalog to predict over
            -1 for all items (from train and val)
            0 for only val items
            n for n (from training catalog) + val items
        n_users_eval: Number of users to evaluate
        include_cold: If `True`, includes unseen items in validation
        cold_only: If `True`, will evaluate only the set of unseen items
        n_xns_as_cold: threshold on the number of interactions an items must 
            have less than to be considered a cold item 
            (typically 0, but some literature uses a nonzero value ex.5)
        seed: seed for random state
    """

    def __init__(self, config, train_data_loader: TrainDataLoader,
                 model_ref: FactModel,
                 limit_items=-1, n_users_eval=200,
                 include_cold=True, cold_only=False, n_xns_as_cold=5,
                 seed: int=0):

        self.model_ref = model_ref
        self.seed = seed
        np.random.seed(self.seed)

        interactions_val = config.get('interactions_val')
        self.user_col_val = interactions_val.user_col
        self.item_col_val = interactions_val.item_col
        self.n_users_eval = n_users_eval

        self.input_fwd_d = None

        self.model = None
        self.metric_ops_d = None
        self.reset_metrics_op = None
        self.eval_ph_d = None

        # Some DS placeholders
        self.ds = None
        self.input_iter = None
        self.input_batch = None

        train_item_counts = train_data_loader.interactions_df.groupby(train_data_loader.item_col).size()
        warm_items = set(train_item_counts.loc[train_item_counts >= n_xns_as_cold].index)

        if include_cold:
            self.cats_d = train_data_loader.cats_d  # .copy() without copy, both objs will be mutated
            self.cats_d_orig = deepcopy(self.cats_d)  # to compare against later
            self.interactions_df, self.user_feats_d, self.item_feats_d = load_simple(
                interactions_val,
                config.get('val_user_features'),
                config.get('val_item_features'),
                config.get('user_specific_feature'),
                config.get('item_specific_feature'),
                self.cats_d,
            )
            if cold_only:
                self.interactions_df = self.interactions_df.loc[
                    ~self.interactions_df[self.item_col_val].isin(warm_items)]

            append_dt_extracts(self.interactions_df,
                               train_data_loader.context_cat_cols,
                               self.cats_d)

            # TODO: same as TrainDataLoader.make_feat_codes()
            # Convert all categorical cols to corresponding codes
            self.user_cat_codes_df = self.user_feats_d[FType.CAT].copy()
            for col in self.user_cat_codes_df.columns:
                self.user_cat_codes_df[col] = self.user_cat_codes_df[col].cat.codes
            self.item_cat_codes_df = self.item_feats_d[FType.CAT].copy()
            for col in self.item_cat_codes_df.columns:
                self.item_cat_codes_df[col] = self.item_cat_codes_df[col].cat.codes

            self.context_cat_codes_df = self.interactions_df[train_data_loader.context_cat_cols].copy()
            for col in self.context_cat_codes_df.columns:
                self.context_cat_codes_df[col] = self.context_cat_codes_df[col].cat.codes

            # Process numerical metadata
            # TODO: assuming numerical features aggregated into 1 table for now
            # ^ else, `self.user_feats_d[FType.NUM]: Iterable`
            self.user_num_feats_df = self.user_feats_d[FType.NUM] \
                if FType.NUM in self.user_feats_d else None
            self.item_num_feats_df = self.item_feats_d[FType.NUM] \
                if FType.NUM in self.item_feats_d else None
            # Gather metadata (size) of num feats
            self.num_meta = {}
            if self.user_num_feats_df is not None:
                self.num_meta['user_num_feats'] = self.user_num_feats_df.shape[1]
            if self.item_num_feats_df is not None:
                self.num_meta['item_num_feats'] = self.item_num_feats_df.shape[1]
            # Concat the training features and dedupe
            # todo: the default case below can be written as concat empty with existing
                # if we wanted to move this concat outside the if statement

            self.user_cat_codes_df = pd.concat(
                [self.user_cat_codes_df, train_data_loader.user_feats_codes_df], axis=0)
            self.user_cat_codes_df = self.user_cat_codes_df[
                ~self.user_cat_codes_df.index.duplicated(keep='last')]
            self.item_cat_codes_df = pd.concat(
                [self.item_cat_codes_df, train_data_loader.item_feats_codes_df], axis=0)
            self.item_cat_codes_df = self.item_cat_codes_df[
                ~self.item_cat_codes_df.index.duplicated(keep='last')]
            if self.user_num_feats_df is not None:
                self.user_num_feats_df = pd.concat(
                    [self.user_num_feats_df, train_data_loader.user_num_feats_df], axis=0)
                self.user_num_feats_df = self.user_num_feats_df[
                    ~self.user_num_feats_df.index.duplicated(keep='last')]
            if self.item_num_feats_df is not None:
                self.item_num_feats_df = pd.concat(
                    [self.item_num_feats_df, train_data_loader.item_num_feats_df], axis=0)
                self.item_num_feats_df = self.item_num_feats_df[
                    ~self.item_num_feats_df.index.duplicated(keep='last')]

            # Get the cold users/items that we need to zero enforce
            self.zero_init_rows = {}
            for col in self.cats_d.keys():
                new_ids = set(self.cats_d[col]) - set(self.cats_d_orig[col])
                id_to_ind_d = dict(zip(self.cats_d[col], range(len(self.cats_d[col]))))
                new_inds = [id_to_ind_d[i] for i in new_ids]  # TODO: I think this is not optimal
                self.zero_init_rows[col] = new_inds

        else:
            self.cats_d = train_data_loader.cats_d
            ##
            self.user_cat_codes_df = train_data_loader.user_feats_codes_df
            self.item_cat_codes_df = train_data_loader.item_feats_codes_df
            self.context_cat_codes_df = train_data_loader.context_feats_codes_df
            self.user_num_feats_df = train_data_loader.user_num_feats_df
            self.item_num_feats_df = train_data_loader.item_num_feats_df
            self.num_meta = train_data_loader.num_meta

            self.interactions_df = load_simple_warm_cats(
                interactions_val,
                self.cats_d[self.user_col_val],
                self.cats_d[self.item_col_val],
            )

            self.interactions_df = self.interactions_df.loc[
                self.interactions_df[self.item_col_val].isin(warm_items)]
            
            self.zero_init_rows = None

        self.user_ids_val = np.array(list(set(
            self.interactions_df[self.user_col_val]
        ).intersection(set(self.user_cat_codes_df.index))))

        self.item_ids = self.cats_d[self.item_col_val].copy()
        if limit_items >= 0:
            # This will consider all "new"/validation items
            #   plus a limited selection of "old"/training items
            #   (no special logic to handle overlapping sets)
            np.random.shuffle(self.item_ids)

            val_item_ids = list(self.interactions_df[self.item_col_val].unique().to_dense())
            self.item_ids = list(set(self.item_ids[:limit_items] + val_item_ids))

        # Re-encode items to only the catalog we care about
        self.interactions_df['item_reenc'] = self.interactions_df[self.item_col_val]\
            .cat.set_categories(self.item_ids)

        logger.info(f'Evaluating on {len(self.item_ids)} items')

        np.random.shuffle(self.user_ids_val)

        with tf.name_scope('placeholders'):
            self.input_fwd_d = self.model_ref.get_fwd_dict(
                batch_size=len(self.item_ids))

    def make_ops(self, model):
        # Eval ops
        # Define our metrics: MAP@10 and AUC
        self.model = model

        self.metric_ops_d, self.reset_metrics_op, self.eval_ph_d = \
            make_metrics_ops(self.model.forward, self.input_fwd_d)

        self.init_ds()

    def init_ds(self):
        def cur_user_fwd_gen():
            # use the same users in the same order if this gen is called again
            pred_feeder_gen = items_pred_dicter_gen(
                self.user_ids_val, self.item_ids,
                self.user_cat_codes_df, self.item_cat_codes_df,
                self.user_num_feats_df, self.item_num_feats_df,
                input_fwd_d=None,
            )
            for user_id, cur_user_fwd_dict in pred_feeder_gen:
                y_true = self.interactions_df \
                    .loc[self.interactions_df[self.user_col_val] == user_id] \
                    ['item_reenc'].cat.codes.values

                y_true_bool = np.zeros(len(self.item_ids), dtype=bool)
                y_true_bool[y_true] = True

                cur_user_fwd_dict['y_true_ph'] = y_true[None, :]
                cur_user_fwd_dict['y_true_bool_ph'] = y_true_bool[None, :]

                yield cur_user_fwd_dict

        input_and_targ_d = {
            **self.input_fwd_d, **self.eval_ph_d,
        }

        self.ds = tf.data.Dataset.from_generator(
            cur_user_fwd_gen,
            {k: v.dtype for k, v in input_and_targ_d.items()},
            {k: v.shape for k, v in input_and_targ_d.items()}, ) \
            .prefetch(10)
        self.input_iter = self.ds.make_initializable_iterator()
        self.input_batch = self.input_iter.get_next()

        # Remake graph replacing placeholders with ds iterator
        self.metric_ops_d, self.reset_metrics_op, self.eval_ph_d = \
            make_metrics_ops(self.model.forward, self.input_batch)

    def run_val(self, sess, summary_writer, step):

        if self.n_users_eval < 0:
            n_users_eval = len(self.user_ids_val)
        else:
            n_users_eval = min(self.n_users_eval, len(self.user_ids_val))

        macro_metrics = defaultdict(lambda: [])
        sess.run(tf.local_variables_initializer())
        sess.run(self.input_iter.initializer)
        sess.run(self.reset_metrics_op)
        for _ in tqdm(range(n_users_eval)):

            # Run updates
            sess.run([tup[1] for tup in self.metric_ops_d.values()])
            # Run and store aggregation
            metric_vals = sess.run(
                [tup[0] for tup in self.metric_ops_d.values()])
            for m, v in zip(self.metric_ops_d.keys(), metric_vals):
                macro_metrics[m].append(v)

            sess.run(self.reset_metrics_op)

        ret_d = {}
        for m, vals in macro_metrics.items():
            metric_score = np.mean(vals)
            metric_score_std = np.std(vals)
            metric_val_summary = tf.Summary(value=[
                tf.Summary.Value(tag=f'{m}_val',
                                 simple_value=metric_score)])
            logger.info(f'(val){m} = {metric_score} +/- {metric_score_std}')
            if summary_writer is not None:
                summary_writer.add_summary(metric_val_summary, step)

            ret_d[m] = metric_score

        return ret_d

    def run_val_context(self, sess, summary_writer, step):
        """ On a per-context level instead of per-user
            Each evaluation instance will likely only have a single positive
            (We assume a user to likely to make purchases in different contexts
            )
        """
        return eval_things_context(
            sess,
            self.interactions_df,
            self.user_col_val, self.item_col_val,
            self.user_ids_val, self.item_ids,
            self.user_cat_codes_df, self.item_cat_codes_df,
            self.user_num_feats_df, self.item_num_feats_df,
            self.input_fwd_d,
            self.metric_ops_d, self.reset_metrics_op, self.eval_ph_d,
            n_xn_eval=self.n_users_eval,
            summary_writer=summary_writer, step=step,
            model=self.model,  # todo: temp

            context_inds=np.arange(len(self.interactions_df)),
            context_cat_codes_df=self.context_cat_codes_df,
        )

