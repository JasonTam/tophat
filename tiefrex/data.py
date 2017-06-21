import pandas as pd
import numpy as np
import tensorflow as tf
from lib_cerebro_py import custom_io
from lib_cerebro_py.log import logger, log_shape_or_npartitions
from tiefrex.core import fwd_dict_via_cats, pair_dict_via_cols
from tiefrex import eval


class TrainDataLoader(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        interactions_train = config.get('train_interactions')
        path_interactions = interactions_train.path
        activity_col = interactions_train.activity_column

        user_col = interactions_train.user_id_column
        self.user_col = user_col

        item_col = interactions_train.item_id_column
        self.item_col = item_col

        activity_filter = interactions_train.filter_activity_set
        item_features = config.get('item_features')
        path_item_features = item_features[0].path

        self.interactions_df, self.user_feats_df, self.item_feats_df = load_simple(
            path_interactions, None, path_item_features,
            user_col, item_col,
            activity_col, activity_filter
        )
        self.user_feat_cols = self.user_feats_df.columns.tolist()
        self.item_feat_cols = self.item_feats_df.columns.tolist()

        self.cats_d = {
            **{feat_name: self.user_feats_df[feat_name].cat.categories.tolist()
               for feat_name in self.user_feat_cols},
            **{feat_name: self.item_feats_df[feat_name].cat.categories.tolist()
               for feat_name in self.item_feat_cols},
        }

        self.user_feats_codes_df = self.user_feats_df.copy()
        for col in self.user_feats_codes_df.columns:
            self.user_feats_codes_df[col] = self.user_feats_codes_df[col].cat.codes
            self.item_feats_codes_df = self.item_feats_df.copy()
        for col in self.item_feats_codes_df.columns:
            self.item_feats_codes_df[col] = self.item_feats_codes_df[col].cat.codes

    def export_data_encoding(self):
        return self.cats_d, self.user_feats_codes_df, self.item_feats_codes_df


class Validator(object):
    def __init__(self, cats_d, user_feats_codes_df, item_feats_codes_df, config):
        self.cats_d = cats_d
        self.user_feats_codes_df = user_feats_codes_df
        self.item_feats_codes_df = item_feats_codes_df

        interactions_val = config.get('eval_interactions')
        path_interactions_val = interactions_val.path
        activity_col_val = interactions_val.activity_column

        self.user_col_val = interactions_val.user_id_column
        self.item_col_val = interactions_val.item_id_column

        activity_filter_val = interactions_val.filter_activity_set

        self.interactions_val_df = load_simple_warm_cats(
            path_interactions_val,
            self.user_col_val, self.item_col_val,
            self.cats_d[self.user_col_val], self.cats_d[self.item_col_val],
            activity_col_val, activity_filter_val,
        )

    def ops(self, model):
        # Eval ops
        # Define our metrics: MAP@10 and AUC
        self.model = model
        self.item_ids = self.cats_d[self.item_col_val]
        self.user_ids_val = self.interactions_val_df[self.user_col_val].unique()
        np.random.shuffle(self.user_ids_val)

        with tf.name_scope('placeholders'):
            self.input_fwd_d = fwd_dict_via_cats(
                self.cats_d.keys(), len(self.item_ids))

        self.metric_ops_d, self.reset_metrics_op, self.eval_ph_d = eval.make_metrics_ops(
            self.model.forward, self.input_fwd_d)

    def run_val(self, sess, summary_writer, step):
        eval.eval_things(sess,
                         self.interactions_val_df,
                         self.user_col_val, self.item_col_val,
                         self.user_ids_val, self.item_ids,
                         self.user_feats_codes_df, self.item_feats_codes_df,
                         self.input_fwd_d,
                         self.metric_ops_d, self.reset_metrics_op, self.eval_ph_d,
                         n_users_eval=20,
                         summary_writer=summary_writer, step=step,
                         )


def load_simple(
        path_interactions,
        path_user_features, path_item_features,
        user_col, item_col,
        activity_col, activity_filter,):
    """Stand-in loader mostly for local testing"""

    interactions_df = custom_io.try_load(path_interactions)
    interactions_df = custom_io.filter_col_isin(
        interactions_df, activity_col, activity_filter).compute()

    if path_user_features:
        user_feats_df = custom_io \
            .try_load(path_user_features, limit_dates=False) \
            .set_index(item_col)
    else:
        user_feats_df = pd.DataFrame(
            index=interactions_df[user_col].drop_duplicates())
    if path_item_features:
        item_feats_df = custom_io \
            .try_load(path_item_features, limit_dates=False) \
            .set_index(item_col)
    else:
        item_feats_df = pd.DataFrame(
            index=interactions_df[item_col].drop_duplicates())

    # Simplifying assumption:
    # All interactions have an entry in the feature dfs
    in_user_feats = interactions_df[user_col].isin(user_feats_df.index)
    in_item_feats = interactions_df[item_col].isin(item_feats_df.index)
    interactions_df = interactions_df.loc[in_user_feats & in_item_feats]

    # And some more filtering
    # Get rid of rows in our feature dataframes that don't show up in interactions
    # (so we dont have a gazillion things in our vocab)
    user_feats_df = user_feats_df.loc[interactions_df[user_col].unique()]
    item_feats_df = item_feats_df.loc[interactions_df[item_col].unique()]

    # Use in the index as a feature (user and item specific eye feature)
    user_feats_df[user_feats_df.index.name] = user_feats_df.index
    item_feats_df[item_feats_df.index.name] = item_feats_df.index

    # Assume all categorical for now
    for col in user_feats_df.columns:
        user_feats_df[col] = user_feats_df[col].astype('category')
    for col in item_feats_df.columns:
        item_feats_df[col] = item_feats_df[col].astype('category')
    interactions_df[user_col] = interactions_df[user_col].astype(
        'category', categories=user_feats_df[user_col].cat.categories)
    interactions_df[item_col] = interactions_df[item_col].astype(
        'category', categories=item_feats_df[item_col].cat.categories)

    log_shape_or_npartitions(interactions_df, 'interactions_df')
    log_shape_or_npartitions(user_feats_df, 'user_feats_df')
    log_shape_or_npartitions(item_feats_df, 'item_feats_df')

    return interactions_df, user_feats_df, item_feats_df


def load_simple_warm_cats(
        path_interactions,
        user_col, item_col,
        users_filt, items_filt,
        activity_col, activity_filter,):
    """Stand-in validation data loader mostly for local testing
        * Filters out new users and items  *

    :param users_filt : typically, existing users
        ex) `user_feats_df[user_col].cat.categories)`
    :param items_filt : typically, existing items
        ex) `item_feats_df[item_col].cat.categories`
    """

    interactions_df = custom_io.try_load(path_interactions)
    interactions_df = custom_io.filter_col_isin(
        interactions_df, activity_col, activity_filter).compute()

    # Simplifying assumption:
    # All interactions have an entry in the feature dfs
    in_user_feats = interactions_df[user_col].isin(users_filt)
    in_item_feats = interactions_df[item_col].isin(items_filt)
    interactions_df = interactions_df.loc[in_user_feats & in_item_feats]

    interactions_df[user_col] = interactions_df[user_col].astype(
        'category', categories=users_filt)
    interactions_df[item_col] = interactions_df[item_col].astype(
        'category', categories=items_filt)

    log_shape_or_npartitions(interactions_df, 'warm interactions_df')

    return interactions_df
