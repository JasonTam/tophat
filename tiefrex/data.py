import pandas as pd
import numpy as np
import tensorflow as tf
from lib_cerebro_py import custom_io
from lib_cerebro_py.log import logger, log_shape_or_npartitions
from typing import Optional, Iterable, Tuple, Dict, List
from collections import defaultdict
from tiefrex.constants import FType


class FeatureSource(object):
    def __init__(self,
                 path: str,
                 feature_type: FType,
                 index_col: Optional[str]=None,
                 use_cols: Optional[List[str]]=None,
                 name=None,
                 ):
        self.name = name
        self.path = path
        self.feature_type = feature_type
        self.index_col = index_col
        self.use_cols = use_cols

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
            if self.use_cols:
                self.data = feat_df[self.use_cols]
            else:
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

        self.interactions_df, self.user_feats_d, self.item_feats_d = load_simple(
            interactions_train,
            config.get('user_features'),
            config.get('item_features'),
            config.get('user_specific_feature'),
            config.get('item_specific_feature'),
        )

        # Process Categorical
        self.user_cat_cols = self.user_feats_d[FType.CAT].columns.tolist()
        self.item_cat_cols = self.item_feats_d[FType.CAT].columns.tolist()

        self.cats_d = {
            **{feat_name: self.user_feats_d[FType.CAT][feat_name].cat.categories.tolist()
               for feat_name in self.user_cat_cols},
            **{feat_name: self.item_feats_d[FType.CAT][feat_name].cat.categories.tolist()
               for feat_name in self.item_cat_cols},
        }

        # Convert all categorical cols to corresponding codes
        self.user_feats_codes_df = self.user_feats_d[FType.CAT].copy()
        for col in self.user_feats_codes_df.columns:
            self.user_feats_codes_df[col] = self.user_feats_codes_df[col].cat.codes
        self.item_feats_codes_df = self.item_feats_d[FType.CAT].copy()
        for col in self.item_feats_codes_df.columns:
            self.item_feats_codes_df[col] = self.item_feats_codes_df[col].cat.codes

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

    def export_data_encoding(self):
        return self.cats_d, self.user_feats_codes_df, self.item_feats_codes_df


def load_simple(
        interactions_src: InteractionsSource,
        user_features_srcs: Optional[Iterable[FeatureSource]],
        item_features_srcs: Optional[Iterable[FeatureSource]],
        user_specific_feature: bool=True,
        item_specific_feature: bool=True,
) -> Tuple[pd.DataFrame, Dict[FType, pd.DataFrame], Dict[FType, pd.DataFrame]]:
    """
    Stand-in loader mostly for local testing
    :param interactions_src: interactions data source
    :param user_features_srcs: user feature data sources
        if `None`, user_id will be the sole user feature
    :param item_features_srcs: item feature data sources
        if `None`, item_id will be the sole item feature
    :param user_specific_feature: if `True`, includes a user_id as a feature
    :param item_specific_feature: if `True`, includes a item_id as a feature
    :return: 
    """

    interactions_df = interactions_src.load().data
    user_col = interactions_src.user_col
    item_col = interactions_src.item_col

    user_feats_d = {}
    item_feats_d = {}
    if user_features_srcs:
        user_feats_d.update(load_many_srcs(user_features_srcs))
    else:
        user_feats_d[FType.CAT] = pd.DataFrame(
            index=interactions_df[user_col].drop_duplicates())
        user_specific_feature = True
    if item_features_srcs:
        item_feats_d.update(load_many_srcs(item_features_srcs))
    else:
        item_feats_d[FType.CAT] = pd.DataFrame(
            index=interactions_df[item_col].drop_duplicates())
        item_specific_feature = True

    # TODO: Extreme simplifying assumption (should be handled better):
    # All interactions have an entry in the feature dfs
    in_user_feats = interactions_df[user_col].isin(user_feats_d[FType.CAT].index)
    in_item_feats = interactions_df[item_col].isin(item_feats_d[FType.CAT].index)
    interactions_df = interactions_df.loc[in_user_feats & in_item_feats]
    # Same assumption with numerical features
    if FType.NUM in user_feats_d:
        in_user_feats = interactions_df[user_col].isin(user_feats_d[FType.NUM].index)
        interactions_df = interactions_df.loc[in_user_feats]
    if FType.NUM in item_feats_d:
        in_item_feats = interactions_df[item_col].isin(item_feats_d[FType.NUM].index)
        interactions_df = interactions_df.loc[in_item_feats]

    # And some more filtering
    # Get rid of rows in our feature dataframes that don't show up in interactions
    # (so we dont have a gazillion things in our vocab)
    user_feats_d[FType.CAT] = user_feats_d[FType.CAT].loc[interactions_df[user_col].unique()]
    item_feats_d[FType.CAT] = item_feats_d[FType.CAT].loc[interactions_df[item_col].unique()]

    # index alignment for numerical features
    if FType.NUM in user_feats_d:
        user_feats_d[FType.NUM] = user_feats_d[FType.NUM] \
            .loc[interactions_df[user_col].unique()]
    if FType.NUM in item_feats_d:
        item_feats_d[FType.NUM] = item_feats_d[FType.NUM] \
            .loc[interactions_df[item_col].unique()]

    # Use in the index as a feature (user and item specific eye feature)
    if user_specific_feature:
        user_feats_d[FType.CAT][user_feats_d[FType.CAT].index.name] = \
            user_feats_d[FType.CAT].index
    if item_specific_feature:
        item_feats_d[FType.CAT][item_feats_d[FType.CAT].index.name] = \
            item_feats_d[FType.CAT].index

    # Cast categorical
    for col in user_feats_d[FType.CAT].columns:
        user_feats_d[FType.CAT][col] = user_feats_d[FType.CAT][col].astype('category')
    for col in item_feats_d[FType.CAT].columns:
        item_feats_d[FType.CAT][col] = item_feats_d[FType.CAT][col].astype('category')
    interactions_df[user_col] = interactions_df[user_col].astype(
        'category', categories=user_feats_d[FType.CAT][user_col].cat.categories)
    interactions_df[item_col] = interactions_df[item_col].astype(
        'category', categories=item_feats_d[FType.CAT][item_col].cat.categories)

    log_shape_or_npartitions(interactions_df, 'interactions_df')
    log_shape_or_npartitions(user_feats_d[FType.CAT], 'user_feats_df')
    log_shape_or_npartitions(item_feats_d[FType.CAT], 'item_feats_df')

    return interactions_df, user_feats_d, item_feats_d


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


def load_many_srcs(features_srcs: Iterable[FeatureSource]):
    src_d = defaultdict(list)
    for feat_src in features_srcs:
        src_d[feat_src.feature_type].append(feat_src.load().data)

    # Upfront join
    # TODO: may consider NOT joining multiple numerical frames upfront
    for feature_type, df_l in src_d.items():
        src_d[feature_type] = pd.concat(df_l, axis=1)
    return src_d
