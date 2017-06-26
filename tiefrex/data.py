import pandas as pd
import numpy as np
import tensorflow as tf
from lib_cerebro_py import custom_io
from lib_cerebro_py.log import logger, log_shape_or_npartitions
from tiefrex.core import fwd_dict_via_cats, pair_dict_via_cols
from tiefrex import evaluation
from typing import Optional
from enum import Enum


class FeatureType(Enum):
    CATEGORICAL = 1
    CONTINUOUS = 2


class FeatureSource(object):
    def __init__(self,
                 path: str,
                 feature_type: FeatureType,
                 index_col: Optional[str]=None,
                 ):
        self.path = path
        self.feature_type = feature_type
        self.index_col = index_col

        self.data = None

    def load(self):
        if self.data is not None:
            logger.info('Already loaded')
        else:
            feat_df = custom_io \
                .try_load(self.path, limit_dates=False)
            if self.index_col:
                feat_df.set_index(self.index_col, inplace=True)
            if hasattr(feat_df, 'compute'):  # cant `.isin` dask
                feat_df = feat_df.compute()
            self.data = feat_df
        return self


class InteractionsSource(object):
    def __init__(self,
                 path: str,
                 user_col: str,
                 item_col: str,
                 activity_col: Optional[str]=None,
                 activity_filter_set: Optional[set]=None,
                 ):
        self.path = path
        self.user_col = user_col
        self.item_col = item_col
        self.activity_col = activity_col
        self.activity_filter_set = activity_filter_set

        self.data = None

    def load(self):
        if self.data is not None:
            logger.info('Already loaded')
        else:
            interactions_df = custom_io.try_load(self.path)
            if self.activity_col and self.activity_filter_set:
                interactions_df = custom_io.filter_col_isin(
                    interactions_df, self.activity_col, self.activity_filter_set)
            if hasattr(interactions_df, 'compute'):
                interactions_df = interactions_df.compute()
            self.data = interactions_df
        return self


class TrainDataLoader(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        interactions_train = config.get('train_interactions')
        self.activity_col = interactions_train.activity_col
        self.user_col = interactions_train.user_col
        self.item_col = interactions_train.item_col

        # TODO: Join on {user|item}_col rather than just taking the head
        user_features = config.get('user_features')
        item_features = config.get('item_features')

        self.interactions_df, self.user_feats_df, self.item_feats_df = load_simple(
            interactions_train,
            user_features[0] if user_features else None,
            item_features[0] if item_features else None,
        )
        self.user_feat_cols = self.user_feats_df.columns.tolist()
        self.item_feat_cols = self.item_feats_df.columns.tolist()

        self.cats_d = {
            **{feat_name: self.user_feats_df[feat_name].cat.categories.tolist()
               for feat_name in self.user_feat_cols},
            **{feat_name: self.item_feats_df[feat_name].cat.categories.tolist()
               for feat_name in self.item_feat_cols},
        }

        # Convert all categorical cols to corresponding codes
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

        self.user_col_val = interactions_val.user_col
        self.item_col_val = interactions_val.item_col

        self.interactions_val_df = load_simple_warm_cats(
            interactions_val,
            self.cats_d[self.user_col_val], self.cats_d[self.item_col_val],
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

        self.metric_ops_d, self.reset_metrics_op, self.eval_ph_d = evaluation.make_metrics_ops(
            self.model.forward, self.input_fwd_d)

    def run_val(self, sess, summary_writer, step):
        evaluation.eval_things(
            sess,
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
        interactions_src: InteractionsSource,
        user_features_src: Optional[FeatureSource],
        item_features_src: Optional[FeatureSource],
):
    """
    Stand-in loader mostly for local testing
    :param interactions_src: interactions data source
    :param user_features_src: user features data source
    :param item_features_src: item features data source
    :return: 
    """

    interactions_df = interactions_src.load().data
    user_col = interactions_src.user_col
    item_col = interactions_src.item_col

    if user_features_src:
        user_feats_df = user_features_src.load().data
    else:
        user_feats_df = pd.DataFrame(
            index=interactions_df[user_col].drop_duplicates())
    if item_features_src:
        item_feats_df = item_features_src.load().data
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
    # TODO: read {user|item}_specific_feature: bool for conditional application of below
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
        interactions_src: InteractionsSource,
        users_filt, items_filt,
):
    """Stand-in validation data loader mostly for local testing
        * Filters out new users and items  *
    :param interactions_src: interactions data source
    :param users_filt : typically, existing users
        ex) `user_feats_df[user_col].cat.categories)`
    :param items_filt : typically, existing items
        ex) `item_feats_df[item_col].cat.categories`
    """

    interactions_df = interactions_src.load().data
    user_col = interactions_src.user_col
    item_col = interactions_src.item_col

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
