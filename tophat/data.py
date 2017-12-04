import os
import pandas as pd
from collections import defaultdict
from lib_cerebro_py import custom_io
from lib_cerebro_py.log import logger, log_shape_or_npartitions
from typing import Optional, Iterable, Tuple, Dict, List, Any, Sized

from tophat.constants import FType
from tophat.utils.pp_utils import append_dt_extracts


class FeatureSource(object):
    """Container for a source of feature-related data

    Args:
        path: Path of the data
        feature_type: Type of feature (ex. categorical, numerical)
        index_col: Name of the column to set as the index
        use_cols: Subset of columns to consider
        name: Name of the data source
    """

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

    def __str__(self):
        return f'FeatureSource({self.name})'

    def __repr__(self):
        return '<%s.%s (%s) at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.path,
            hex(id(self))
        )

    def load(self, force_reload=False):
        if not (force_reload or self.data is None):
            logger.info('Already loaded')
        else:
            feat_df = custom_io \
                .try_load(self.path, limit_dates=False)
            if hasattr(feat_df, 'compute'):  # cant `.isin` dask
                feat_df = feat_df.compute()
            if self.index_col:
                feat_df.set_index(self.index_col, inplace=True)
            if self.use_cols:
                self.data = feat_df[self.use_cols]
            elif self.use_cols is not None and self.use_cols == []:
                # Empty dataframe (rely on {user|item} specific feature
                self.data = pd.DataFrame(index=feat_df.index)
            else:
                # Use entire dataframe
                self.data = feat_df
        return self


class InteractionsSource(object):
    """Container for a source of interaction-related data

    Args:
        path: Path of the data
        user_col: Name of the user column
        item_col: Name of the item column
        activity_col: Name of the interaction type
        activity_filter_set: Subset of interaction types to consider
        assign_dates: If True, and loading from date-partitioned
            directories, assign the date information to a column
        days_lookback: Max number of days to look back from today
        date_lookforward: Furthest (most recent) date to consider
    """

    def __init__(self,
                 path: str,
                 user_col: str,
                 item_col: str,
                 activity_col: Optional[str]=None,
                 activity_filter_set: Optional[set]=None,
                 assign_dates: bool=True,
                 days_lookback: int=9999,
                 date_lookforward: Optional[str]=None
                 ):

        self.path = path
        self.user_col = user_col
        self.item_col = item_col
        self.activity_col = activity_col
        self.activity_filter_set = activity_filter_set
        self.assign_dates = assign_dates
        self.days_lookback = days_lookback
        self.date_lookforward = date_lookforward

        self.data = None

    def __str__(self):
        return f'InteractionsSource({self.path})'

    def __repr__(self):
        return '<%s.%s (%s) at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.path,
            hex(id(self))
        )

    def load(self):
        if self.data is not None:
            logger.info('Already loaded')
        else:
            if os.path.splitext(self.path)[-1]:
                # single file -- can't selectively read partitions by date
                interactions_df = custom_io.try_load(
                    self.path,
                    limit_dates=False)
            else:
                interactions_df = custom_io.try_load(
                    self.path,
                    limit_dates=True,
                    days_lookback=self.days_lookback,
                    date_lookforward=self.date_lookforward,
                    assign_dates=self.assign_dates,
                )
            if 'value' in interactions_df.columns \
                    and self.item_col not in interactions_df.columns:
                interactions_df = interactions_df.rename(
                    columns={'value': self.item_col})
            if self.activity_col and self.activity_filter_set:
                interactions_df = custom_io.filter_col_isin(
                    interactions_df,
                    self.activity_col, self.activity_filter_set)
            if hasattr(interactions_df, 'compute'):
                interactions_df = interactions_df.compute()
            self.data = interactions_df
        return self


class TrainDataLoader(object):
    """Convenience container to load and preprocess various sources of
    training data

    Args:
        interactions_train: training interactions source
        user_features: user feature sources
        item_features: item feature sources
        user_specific_feature: include the user-specific feature
        item_specific_feature: include the item-specific feature
        context_cols: columns to consider interaction context
        batch_size: batch size
    """

    def __init__(self,
                 interactions_train: InteractionsSource,
                 user_features: Optional[Iterable[FeatureSource]],
                 item_features: Optional[Iterable[FeatureSource]],
                 user_specific_feature: bool=True,
                 item_specific_feature: bool=True,
                 context_cols: Optional[Iterable[str]]=None,
                 batch_size: int=128,
                 ):

        self.batch_size = batch_size
        self.activity_col = interactions_train.activity_col
        self.user_col = interactions_train.user_col
        self.item_col = interactions_train.item_col

        self.user_feats_codes_df = None
        self.item_feats_codes_df = None
        self.context_feats_codes_df = None
        self.user_num_feats_df = None
        self.item_num_feats_df = None
        self.num_meta = None

        self.interactions_df, self.user_feats_d, self.item_feats_d = \
            load_simple(
                interactions_train,
                user_features,
                item_features,
                user_specific_feature,
                item_specific_feature,
            )

        # Process Categorical
        self.user_cat_cols = self.user_feats_d[FType.CAT].columns.tolist()
        self.item_cat_cols = self.item_feats_d[FType.CAT].columns.tolist()

        self.cats_d = {
            **{feat_name: self.user_feats_d[FType.CAT][feat_name]
                .cat.categories.tolist()
               for feat_name in self.user_cat_cols},
            **{feat_name: self.item_feats_d[FType.CAT][feat_name]
                .cat.categories.tolist()
               for feat_name in self.item_cat_cols},
        }
        # If the user or item ids are not present in feature tables
        if self.user_col not in self.cats_d:
            self.cats_d[self.user_col] = self.interactions_df[self.user_col]\
                .cat.categories.tolist()
        if self.item_col not in self.cats_d:
            self.cats_d[self.item_col] = self.interactions_df[self.item_col]\
                .cat.categories.tolist()

        self.context_cat_cols = context_cols or []
        if self.context_cat_cols:
            append_dt_extracts(self.interactions_df, self.context_cat_cols)
            for col in self.context_cat_cols:
                self.cats_d[col] = self.interactions_df[col]\
                    .cat.categories.tolist()

        self.make_feat_codes()
        self.process_num()

    def export_data_encoding(self):
        return self.cats_d, self.user_feats_codes_df, self.item_feats_codes_df

    def make_feat_codes(self):
        # Convert all categorical cols to corresponding codes
        self.user_feats_codes_df = self.user_feats_d[FType.CAT].copy()
        for col in self.user_feats_codes_df.columns:
            self.user_feats_codes_df[col] = self.user_feats_codes_df[col]\
                .cat.codes
        self.item_feats_codes_df = self.item_feats_d[FType.CAT].copy()
        for col in self.item_feats_codes_df.columns:
            self.item_feats_codes_df[col] = self.item_feats_codes_df[col]\
                .cat.codes

        if self.context_cat_cols:
            self.context_feats_codes_df = \
                self.interactions_df[self.context_cat_cols].copy()
            for col in self.context_feats_codes_df.columns:
                self.context_feats_codes_df[col] = \
                    self.context_feats_codes_df[col].cat.codes

    def process_num(self):
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


def cast_cat(feats_d: Dict[FType, pd.DataFrame],
             existing_cats_d: Optional[Dict[str, Sized]]=None
             ) -> Dict[FType, pd.DataFrame]:
    """Casts feature columns to categorical
    -- optionally applying existing categories

    Args:
        feats_d: Dictionary of feature dataframes
        existing_cats_d: Optional dictionary of existing categories

    Returns:
        Modified version of `feats_d`
    """

    for col in feats_d[FType.CAT].columns:
        if existing_cats_d and col in existing_cats_d:
            # todo: shouldnt we be adding the new cats?
            existing_cats_d[col] += list(
                set(feats_d[FType.CAT][col].unique()) -
                set(existing_cats_d[col]))
            existing_cats = existing_cats_d[col]
        else:
            existing_cats = None
        feats_d[FType.CAT][col] = feats_d[FType.CAT][col].astype(
            'category', categories=existing_cats)
    return feats_d


def load_simple(
        interactions_src: InteractionsSource,
        user_features_srcs: Optional[Iterable[FeatureSource]],
        item_features_srcs: Optional[Iterable[FeatureSource]],
        user_specific_feature: bool=True,
        item_specific_feature: bool=True,
        existing_cats_d: Optional[Dict[str, List[Any]]]=None,
) -> Tuple[pd.DataFrame, Dict[FType, pd.DataFrame], Dict[FType, pd.DataFrame]]:
    """Stand-in loader mostly for local testing

    Args:
        interactions_src: Interactions data source
        user_features_srcs: User feature data source(s)
            if `None`, user_id will be the sole user feature
        item_features_srcs: Item feature data source(s)
            if `None`, item_id will be the sole item feature
        user_specific_feature: If `True`, includes a user_id as a feature
        item_specific_feature: If `True`, includes a item_id as a feature
        existing_cats_d: Optional dictionary of existing categories

    Returns:
        Tuple of preprocessed interactions, user features, and item_features
    """

    interactions_df = interactions_src.load().data
    user_col = interactions_src.user_col
    item_col = interactions_src.item_col

    user_feats_d = {}
    item_feats_d = {}
    if user_features_srcs:
        user_feats_d.update(load_many_srcs(user_features_srcs))
    if not user_features_srcs or not any([src.feature_type == FType.CAT
                                          for src in user_features_srcs]):
        user_feats_d[FType.CAT] = pd.DataFrame(
            index=interactions_df[user_col].drop_duplicates())
        user_specific_feature = True
    if item_features_srcs:
        item_feats_d.update(load_many_srcs(item_features_srcs))
    if not item_features_srcs or not any([src.feature_type == FType.CAT
                                          for src in item_features_srcs]):
        item_feats_d[FType.CAT] = pd.DataFrame(
            index=interactions_df[item_col].drop_duplicates())
        item_specific_feature = True

    # TODO: Extreme simplifying assumption (should be handled better):
    #     This gets rid of missing side features
    user_feats_d[FType.CAT].dropna(axis=0, inplace=True)
    item_feats_d[FType.CAT].dropna(axis=0, inplace=True)

    # TODO: Another simplifying assumption:
    interactions_df, user_feats_d, item_feats_d, = simplifying_assumption(
        interactions_df,
        user_feats_d, item_feats_d,
        user_col, item_col,
    )

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
    user_feats_d = cast_cat(user_feats_d, existing_cats_d)
    item_feats_d = cast_cat(item_feats_d, existing_cats_d)

    existing_user_cats = user_feats_d[FType.CAT][user_col].cat.categories\
        if user_col in user_feats_d[FType.CAT] else None
    interactions_df[user_col] = interactions_df[user_col].astype(
        'category', categories=existing_user_cats)

    existing_item_cats = item_feats_d[FType.CAT][item_col].cat.categories\
        if item_col in item_feats_d[FType.CAT] else None
    interactions_df[item_col] = interactions_df[item_col].astype(
        'category', categories=existing_item_cats)

    log_shape_or_npartitions(interactions_df, 'interactions_df')
    log_shape_or_npartitions(user_feats_d[FType.CAT], 'user_feats_df')
    log_shape_or_npartitions(item_feats_d[FType.CAT], 'item_feats_df')

    return interactions_df, user_feats_d, item_feats_d


def simplifying_assumption(interactions_df,
                           user_feats_d, item_feats_d,
                           user_col, item_col,
                           ):
    """OUTOFPLACE:
    filtering to make sure we only have known interaction users/items
    """
    # All interactions have an entry in the feature dfs
    in_user_feats = interactions_df[user_col].isin(
        user_feats_d[FType.CAT].index)
    in_item_feats = interactions_df[item_col].isin(
        item_feats_d[FType.CAT].index)
    interactions_df = interactions_df.loc[in_user_feats & in_item_feats]
    # Same assumption with numerical features
    if FType.NUM in user_feats_d:
        in_user_feats = interactions_df[user_col].isin(
            user_feats_d[FType.NUM].index)
        interactions_df = interactions_df.loc[in_user_feats]
    if FType.NUM in item_feats_d:
        in_item_feats = interactions_df[item_col].isin(
            item_feats_d[FType.NUM].index)
        interactions_df = interactions_df.loc[in_item_feats]

    # And some more filtering
    # Get rid of rows in feature df that don't show up in interactions
    # (so we dont have a gazillion things in our vocab)
    user_feats_d[FType.CAT] = user_feats_d[FType.CAT]\
        .loc[interactions_df[user_col].unique()]
    item_feats_d[FType.CAT] = item_feats_d[FType.CAT]\
        .loc[interactions_df[item_col].unique()]

    return interactions_df, user_feats_d, item_feats_d,


def load_simple_warm_cats(
        interactions_src: InteractionsSource,
        users_filt: Optional[Iterable]=None,
        items_filt: Optional[Iterable]=None,
) -> pd.DataFrame:
    """Stand-in validation data loader mostly for local testing

    Args:
        interactions_src: Interactions data source
        users_filt: Subset of users to consider (typically, existing users)
            ex) `user_feats_df[user_col].cat.categories)`
        items_filt: Subset of items to consider (typically, existing items)
            ex) `item_feats_df[item_col].cat.categories`

    Returns:
        Preprocessed interaction dataframe
    """

    interactions_df = interactions_src.load().data
    user_col = interactions_src.user_col
    item_col = interactions_src.item_col

    filt_s = pd.Series([True]*len(interactions_df),
                       index=interactions_df.index)

    # All interactions should have an entry in provided filters
    if users_filt is not None:
        filt_s &= interactions_df[user_col].isin(users_filt)
    if items_filt is not None:
        filt_s &= interactions_df[item_col].isin(items_filt)

    interactions_df = interactions_df.loc[filt_s].copy()

    interactions_df[user_col] = interactions_df[user_col].astype(
        'category', categories=users_filt)
    interactions_df[item_col] = interactions_df[item_col].astype(
        'category', categories=items_filt)

    log_shape_or_npartitions(interactions_df, 'warm interactions_df')

    return interactions_df


def load_many_srcs(features_srcs: Iterable[FeatureSource]):
    """Load and concatenate many feature sources
    
    Args:
        features_srcs: Feature sources

    Returns:

    """
    src_d = defaultdict(list)
    for feat_src in features_srcs:
        src_d[feat_src.feature_type].append(feat_src.load().data)

    # Upfront join
    # TODO: may consider NOT joining multiple numerical frames upfront
    for feature_type, df_l in src_d.items():
        src_d[feature_type] = pd.concat(df_l, axis=1)
    return src_d
