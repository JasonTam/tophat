# TODO: This whole file needs some serious refactoring

from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Optional

from tophat.constants import FType, FGroup
from tophat.data import (load_simple_warm_cats, load_simple,
                         InteractionsSource, FeatureSourceDictType)
from tophat.evaluation.metrics import make_metrics_ops
from tophat.evaluation.transport import (
    items_pred_dicter_gen, items_pred_dicter_gen_context)
from tophat.tasks.factorization import FactorizationTask
from tophat.tasks.wrapper import FactorizationTaskWrapper
from tophat.utils.log import logger
from tophat.utils.pp_utils import append_dt_extracts


class Validator(object):
    """Convenience validation object with various book-keeping
    
    Args:
        config:
        parent_task_wrapper: task to reference from
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

    def __init__(self, interactions_val_src: InteractionsSource,
                 parent_task_wrapper: FactorizationTaskWrapper,
                 limit_items=-1, n_users_eval=200,
                 include_cold=True, cold_only=False, n_xns_as_cold=5,
                 features_srcs: Optional[FeatureSourceDictType] = None,
                 specific_feature: Optional[Dict[FGroup, bool]] = None,
                 seed: int=0,
                 name: Optional[str] = None,
                 ):

        self.name = name or ''
        self.parent_task_wrapper = parent_task_wrapper
        train_data_loader = parent_task_wrapper.data_loader
        self.model_ref: FactorizationTask = None
        self.rand = np.random.RandomState(seed)

        self.user_col_val = interactions_val_src.user_col
        self.item_col_val = interactions_val_src.item_col
        self.n_users_eval = n_users_eval

        # Allocate processed interactions and features
        self.cats_d = None
        self.cat_codes_dfs = None
        self.num_feats_dfs = None
        self.num_meta = None
        self.interactions_df = None
        self.zero_init_rows = None

        # Allocate Operations
        self.input_fwd_d: Dict[str, tf.Tensor] = None

        self.metric_ops_d = None
        self.reset_metrics_op = None
        self.eval_ph_d = None

        # Allocate dataset stuff
        self.ds = None
        self.input_iter = None
        self.input_batch = None

        # If an item occurs in training less than `n_xns_as_cold` times,
        # it is considered a cold item; otherwise, warm
        train_item_counts = train_data_loader.interactions_df\
            .groupby(train_data_loader.item_col).size()
        warm_items = set(
            train_item_counts.loc[train_item_counts >= n_xns_as_cold].index)

        if include_cold:
            self.init_cold(train_data_loader, interactions_val_src, warm_items,
                           features_srcs, specific_feature,
                           cold_only)

        else:
            self.init_warm(train_data_loader, interactions_val_src, warm_items)

        self.user_ids_val = np.array(list(set(
            self.interactions_df[self.user_col_val]
        ).intersection(set(self.cat_codes_dfs[FGroup.USER].index))))

        # TODO: could be less sketchy (esp considering the cold stuff above^)
        # self.item_ids = self.cats_d[self.item_col_val].copy()
        self.item_ids = self.cat_codes_dfs[FGroup.ITEM].index.tolist()

        if limit_items >= 0:
            # This will consider all "new"/validation items
            #   plus a limited selection of "old"/training items
            #   (no special logic to handle overlapping sets)
            self.rand.shuffle(self.item_ids)

            val_item_ids = list(
                self.interactions_df[self.item_col_val].unique().to_dense())
            self.item_ids = list(
                set(self.item_ids[:limit_items] + val_item_ids))

        # Re-encode items to only the catalog we care about
        self.interactions_df['item_reenc'] = self.interactions_df\
            [self.item_col_val].cat.set_categories(self.item_ids)

        logger.info(f'Evaluating on {len(self.item_ids)} items')

        self.rand.shuffle(self.user_ids_val)

    def init_warm(self, train_data_loader, interactions_val_src, warm_items):
        self.cats_d = train_data_loader.cats_d

        self.cat_codes_dfs = train_data_loader.feats_codes_df
        self.num_feats_dfs = train_data_loader.num_feats_df
        self.num_meta = train_data_loader.num_meta

        self.interactions_df = load_simple_warm_cats(
            interactions_val_src,
            self.cats_d[self.user_col_val],
            self.cats_d[self.item_col_val],
        )

        self.interactions_df = self.interactions_df.loc[
            self.interactions_df[self.item_col_val].isin(warm_items)]

        self.zero_init_rows = None

    def init_cold(self, train_data_loader, interactions_val_src, warm_items,
                  features_srcs, specific_feature,
                  cold_only=False,
                  ):
        # Pointer Ref, so both objs will be mutated
        self.cats_d = train_data_loader.cats_d
        self.cats_d_orig = deepcopy(self.cats_d)  # to compare against later
        self.interactions_df, feats_by_group = load_simple(
            interactions_val_src,
            features_srcs,
            specific_feature,
            existing_cats_d=self.cats_d,
        )

        if cold_only:
            self.interactions_df = self.interactions_df.loc[
                ~self.interactions_df[self.item_col_val].isin(warm_items)]

        append_dt_extracts(self.interactions_df,
                           train_data_loader.context_cat_cols,
                           self.cats_d)

        # TODO: same as TrainDataLoader.make_feat_codes()
        # Convert all categorical cols to corresponding codes
        self.cat_codes_dfs = {}
        self.num_feats_dfs = {}
        self.num_meta = {}
        for fgroup, feats_d in feats_by_group.items():

            # Prep cat codes
            cat_code_df = feats_d[FType.CAT].copy()
            for col in cat_code_df.columns:
                cat_code_df[col] = cat_code_df[col].cat.codes

            # Prep num feats
            # TODO: assuming numerical features aggregated into 1 table for now
            # ^ else, `self.user_feats_d[FType.NUM]: Iterable`
            num_feats_df = feats_d[FType.NUM] if FType.NUM in feats_d else None
            # Gather metadata (size) of num feats
            if num_feats_df is not None:
                meta_key = f'{fgroup}_num_feats'
                self.num_meta[meta_key] = num_feats_df.shape[1]

            # Concat the training features and de-dupe
            # todo: default case below can be written as concat empty with existing
            # if we wanted to move this concat outside the if statement
            cat_code_df = pd.concat([
                cat_code_df,
                train_data_loader.feats_codes_df[fgroup]
            ], axis=0)
            cat_code_df = cat_code_df[
                ~cat_code_df.index.duplicated(keep='last')]

            if num_feats_df is not None:
                num_feats_df = pd.concat([
                    num_feats_df,
                    train_data_loader.num_feats_df[fgroup]
                ], axis=0)
                num_feats_df = num_feats_df[
                    ~num_feats_df.index.duplicated(keep='last')]

            # Store the processed dfs
            self.cat_codes_dfs[fgroup] = cat_code_df
            self.num_feats_dfs[fgroup] = num_feats_df

        # Special processing for context
        if train_data_loader.context_cat_cols:
            self.cat_codes_dfs[FGroup.CONTEXT] = self.interactions_df[
                train_data_loader.context_cat_cols].copy()
            for col in self.cat_codes_dfs[FGroup.CONTEXT].columns:
                self.cat_codes_dfs[FGroup.CONTEXT][col] = self.cat_codes_dfs\
                    [FGroup.CONTEXT][col].cat.codes

        # Get the cold users/items that we need to zero enforce
        self.zero_init_rows = {}
        for col in self.cats_d.keys():
            new_ids = set(self.cats_d[col]) - set(self.cats_d_orig[col])
            id_to_ind_d = dict(
                zip(self.cats_d[col], range(len(self.cats_d[col]))))
            new_inds = [id_to_ind_d[i] for i in
                        new_ids]  # TODO: I think this is not optimal
            self.zero_init_rows[col] = new_inds

    def make_ops(self):
        # Eval ops
        # Define our metrics: MAP@10 and AUC
        if not self.parent_task_wrapper.built:
            self.parent_task_wrapper.build()
        self.model_ref = self.parent_task_wrapper.task
        with tf.name_scope('placeholders'):
            # TODO: can we just use model.get_fwd_dict? whats with model_ref?
            self.input_fwd_d = self.model_ref.get_fwd_dict(
                batch_size=len(self.item_ids))

        self.metric_ops_d, self.reset_metrics_op, self.eval_ph_d = \
            make_metrics_ops(self.model_ref.forward, self.input_fwd_d)

        self.init_ds()

    def init_ds(self):
        def cur_user_fwd_gen():
            # use the same users in the same order if this gen is called again
            pred_feeder_gen = items_pred_dicter_gen(
                self.user_ids_val, self.item_ids,
                self.cat_codes_dfs,
                self.num_feats_dfs,
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
            make_metrics_ops(self.model_ref.forward, self.input_batch)

    def run_val(self, sess, summary_writer=None, step=None, macro=False):
        """

        Args:
            sess:
            summary_writer:
            step: step for summary writer
            macro: Macro average across users, else, micro average across
                interactions

        Returns:

        """
        if self.metric_ops_d is None:
            logger.info('ops missing, making them now via `self.make_ops`')
            self.make_ops()

        if self.n_users_eval < 0:
            n_users_eval = len(self.user_ids_val)
        else:
            n_users_eval = min(self.n_users_eval, len(self.user_ids_val))

        metrics_per_user = defaultdict(lambda: [])
        sess.run(tf.local_variables_initializer())
        sess.run(self.input_iter.initializer)
        sess.run(self.reset_metrics_op)
        for _ in tqdm(range(n_users_eval)):

            # Run updates
            sess.run([tup[1] for tup in self.metric_ops_d.values()])
            # Run and store aggregation
            metric_vals = sess.run(
                [tup[0] for tup in self.metric_ops_d.values()])

            if macro:
                for m, v in zip(self.metric_ops_d.keys(), metric_vals):
                    metrics_per_user[m].append(v)

                # Reset for each user for macro metrics
                sess.run(self.reset_metrics_op)
        # Micro agg will just be the last updated value (without resets)
        micro_metrics = dict(zip(self.metric_ops_d.keys(), metric_vals))

        ret_d = {}
        for m in self.metric_ops_d.keys():
            if macro:
                vals = metrics_per_user[m]
                metric_score = np.mean(vals)
                metric_score_std = np.std(vals)
                logger.info(
                    f'(val){m} = {metric_score} +/- {metric_score_std}')
            else:
                metric_score = micro_metrics[m]
                logger.info(f'(val){m} = {metric_score}')

            metric_val_summary = tf.Summary(value=[
                tf.Summary.Value(tag=f'{self.name}/{m}_val',
                                 simple_value=metric_score)])
            if summary_writer is not None:
                summary_writer.add_summary(metric_val_summary, step)

            ret_d[m] = metric_score

        return ret_d

    def run_val_context(self, sess, summary_writer=None, step=None,):
        """ On a per-context level instead of per-user
            Each evaluation instance will likely only have a single positive
            (We assume a user to likely to make purchases in different contexts
            )
        """

        """
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
        #         tf.Summary.Value(tag=f'{self.name}/{m}_val',
        #                          simple_value=metric_score)])
        #     logger.info(f'(val){m} = {metric_score}')
        #     if summary_writer is not None:
        #         summary_writer.add_summary(metric_val_summary, step)

        for m, vals in macro_metrics.items():
            metric_score = np.mean(vals)
            metric_score_std = np.std(vals)
            metric_val_summary = tf.Summary(value=[
                tf.Summary.Value(tag=f'{self.name}/{m}_val',
                                 simple_value=metric_score)])
            logger.info(f'(val){m} = {metric_score} +/- {metric_score_std}')
            if summary_writer is not None:
                summary_writer.add_summary(metric_val_summary, step)

        return macro_metrics
        """

        # TODO: Above is some legacy code. Taken out of support
        raise NotImplementedError

