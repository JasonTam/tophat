import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import itertools as it
from collections import defaultdict
from typing import Optional, Iterable, Tuple, Dict, List, Any, Sized, \
    Sequence, Union, Callable

from tophat.constants import FType, FGroup
from tophat.utils.pp_utils import append_dt_extracts
from tophat.utils.convenience import filter_col_isin, log_shape_or_npartitions
from tophat.utils.log import logger


class FeatureSource(object):
    """Container for a source of feature-related data

    Args:
        path: Path of the data
        feature_type: Type of feature (ex. categorical, numerical)
        index_col: Name of the column to set as the index
        use_cols: Subset of columns to consider
        concat_cols: list-like of list-like columns to concat together
        drop_cols: list of columns to drop
        load_fn: function to load features from path
        load_kwargs: kwargs for `load_fn`
        name: Name of the data source
    """

    def __init__(self,
                 path: Union[str, pd.DataFrame],
                 feature_type: FType,
                 index_col: Optional[str]=None,
                 use_cols: Optional[List[str]]=None,
                 concat_cols: Optional[List[Sequence[str]]]=None,
                 drop_cols: Optional[Iterable[str]]=None,
                 load_fn: Optional[Callable[
                     [Union[str, pd.DataFrame]], pd.DataFrame]] = None,
                 load_kwargs: Optional[Dict] = None,
                 name=None,
                 ):

        self.name = name
        self.path = path
        self.load_fn = load_fn or (lambda x: x)
        self.load_kwargs: Dict = load_kwargs or {}
        self.feature_type = feature_type
        self.index_col = index_col
        self.use_cols = use_cols
        self.concat_cols = concat_cols
        self.drop_cols = drop_cols

        self.data = None

    def __str__(self):
        return f'FeatureSource({self.name})'

    def __repr__(self):
        src_name = self.path if isinstance(self.path, str) else 'pre-loaded'
        return '<%s.%s (%s) at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            src_name,
            hex(id(self))
        )

    def load(self, force_reload=False):
        if not (force_reload or self.data is None):
            logger.info('Already loaded')
        else:
            feat_df = self.load_fn(self.path, **self.load_kwargs)
            if hasattr(feat_df, 'compute'):  # cant `.isin` dask
                feat_df = feat_df.compute()
            if self.index_col:
                feat_df.set_index(self.index_col, inplace=True)
                duplicates = feat_df.index.duplicated(keep='last')
                feat_df = feat_df.loc[~duplicates]
            if self.use_cols:
                self.data = feat_df[self.use_cols]
            elif self.use_cols is not None and self.use_cols == []:
                # Empty dataframe (rely on {user|item} specific feature
                self.data = pd.DataFrame(index=feat_df.index)
            else:
                # Use entire dataframe
                self.data = feat_df

            if self.concat_cols is not None:
                self.data = combine_cols(df=self.data,
                                         cols_seq=self.concat_cols)

            if self.drop_cols:
                self.data.drop(list(set(self.drop_cols)), axis=1, inplace=True)

        return self


FeatureSourceDictType = Dict[FGroup, Optional[Iterable[FeatureSource]]]


def combine_cols(df: pd.DataFrame,
                 cols_seq: Sequence[Sequence[str]],
                 sep: str='__'):
    """Concatenates columns (will output str dtypes)
    
    Args:
        df: dataframe to operate on
        cols_seq: list-like of list-like columns to concat together
        sep: string separator
            Note: careful - tensorflow needs `[A-Za-z0-9_.\\-/]*` for scope

    Returns:
        df: df with concatenated columns

    """

    for cols in cols_seq:
        new_col_name = sep.join(cols)
        df[new_col_name] = df[cols[0]].astype(str).str.cat(
            [df[col].astype(str) for col in cols[1:]], sep=sep)

    return df


class InteractionsSource(object):
    """Container for a source of interaction-related data

    Args:
        path: Path of the data or a pre-loaded dataframe
        user_col: Name of the user column
        item_col: Name of the item column
        count_col: Name of the count column
        activity_col: Name of the interaction type
        activity_filter_set: Subset of interaction types to consider
        load_fn: function to load interactions from path
        load_kwargs: kwargs for `load_fn`
        name: name for this object
    """

    def __init__(self,
                 path: Union[str, pd.DataFrame],
                 user_col: str,
                 item_col: str,
                 count_col: Optional[str] = None,
                 activity_col: Optional[str] = None,
                 activity_filter_set: Optional[set] = None,
                 load_fn: Optional[Callable[
                     [Union[str, pd.DataFrame]], pd.DataFrame]] = None,
                 load_kwargs: Optional[Dict] = None,
                 name: Optional[str] = None,
                 ):
        self.name = name or ''
        self.path = path
        self.load_fn = load_fn or (lambda x: x)
        self.load_kwargs: Dict = load_kwargs or {}
        self.user_col = user_col
        self.item_col = item_col
        self.count_col = count_col
        self.activity_col = activity_col
        self.activity_filter_set = activity_filter_set

        self.data = None

    def __str__(self):
        return f'InteractionsSource({self.path})'

    def __repr__(self):
        src_name = self.path if isinstance(self.path, str) else 'pre-loaded'
        return '<%s.%s (%s) at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            src_name,
            hex(id(self))
        )

    def load(self):
        if self.data is not None:
            logger.info('Already loaded')
        else:
            interactions_df = self.load_fn(self.path, **self.load_kwargs)
            if 'value' in interactions_df.columns \
                    and self.item_col not in interactions_df.columns:
                interactions_df = interactions_df.rename(
                    columns={'value': self.item_col})
            if self.activity_col and self.activity_filter_set:
                interactions_df = filter_col_isin(
                    interactions_df,
                    self.activity_col, self.activity_filter_set)
            if hasattr(interactions_df, 'compute'):
                interactions_df = interactions_df.compute()
            self.data = interactions_df
        return self


class InteractionsDerived(object):
    """Container for interaction-related data derived from another
    interaction dataset

    Args:
        xn_parent: the parent interaction source to derive data from
        fn: the function to apply to the parent data
        user_col: optional user column name (if different from parent)
        item_col: optional item column name (if different from parent)
        activity_col: optional activity column name (if different from parent)
        memoize: if True, memoize the derived data, else, apply the function
            on every property call

    Todo:
        Wish we could subclass from `InteractionsSource`, but overriding
        and attribute with a property is messed up

    """

    def __init__(self,
                 xn_parent: InteractionsSource,
                 fn: Callable[[InteractionsSource], pd.DataFrame],
                 user_col: Optional[str]=None,
                 item_col: Optional[str]=None,
                 activity_col: Optional[str]=None,
                 count_col: Optional[str]=None,
                 memoize: bool=True,
                 name: Optional[str] = None,
                 ):

        self.xn_parent = xn_parent
        self.fn = fn
        self.memoize = memoize
        self.memo = None

        self.name = name or f'{xn_parent.name}_child'
        self.user_col = user_col or xn_parent.user_col
        self.item_col = item_col or xn_parent.item_col
        self.activity_col = activity_col or xn_parent.activity_col
        self.count_col = count_col or xn_parent.count_col

    @property
    def data(self):
        if self.memo is None or not self.memoize:
            self.memo = self.fn(self.xn_parent)
        return self.memo

    def load(self):
        self.xn_parent.load()
        return self


class TrainDataLoader(object):
    """Convenience container to load and preprocess various sources of
    training data

    Args:
        interactions_train: training interactions source
        group_features: feature sources keyed by group (user, item)
        specific_feature: include the primary_id-specific feature
        context_cols: columns to consider interaction context
        batch_size: batch size
        existing_cats_d: Optional dictionary of existing categories
        add_new_cats: if `True`, will append newly seen categories to
            book-keeping dictionary of categories (mutates inplace)
    """

    def __init__(self,
                 interactions_train: InteractionsSource,
                 group_features: Optional[FeatureSourceDictType] = None,
                 specific_feature: Optional[Dict[FGroup, bool]] = None,
                 context_cols: Optional[Iterable[str]] = None,
                 batch_size: int=128,
                 existing_cats_d: Optional[Dict[str, List[Any]]] = None,
                 add_new_cats: Optional[bool] = False,
                 name: Optional[str]=None,
                 ):
        self.name = name or interactions_train.name or ''
        self.batch_size = batch_size
        self.activity_col = interactions_train.activity_col
        self.cols = {
            FGroup.USER: interactions_train.user_col,
            FGroup.ITEM: interactions_train.item_col,
            'activity': interactions_train.activity_col,
            'count': interactions_train.count_col,
        }
        group_features = group_features or {
            FGroup.USER: [],
            FGroup.ITEM: [],
        }

        self.feats_codes_df: Dict[FGroup, pd.DataFrame] = {}
        self.num_feats_df: Dict[FGroup, pd.DataFrame] = {}
        self.num_meta = None

        if specific_feature is None:
            specific_feature = defaultdict(lambda: True)

        self.cat_cols = {}
        self.cats_d = existing_cats_d or {}

        self.interactions_df, self.feats_by_group = \
            load_simple(
                interactions_train,
                group_features,
                specific_feature,
                existing_cats_d=self.cats_d,
                add_new_cats=add_new_cats,
            )

        for fgroup in [FGroup.USER, FGroup.ITEM]:
            col = self.cols[fgroup]
            feats = self.feats_by_group[fgroup]
            self.cat_cols[fgroup] = feats[FType.CAT].columns.tolist()

            if not existing_cats_d:
                self.cats_d.update({
                    feat_name: feats[FType.CAT][feat_name].cat.categories.tolist()
                    for feat_name in self.cat_cols[fgroup]
                })

                # If the user or item ids are not present in feature tables
                if col not in self.cats_d:
                    self.cats_d[col] = self.interactions_df[col]\
                        .cat.categories.tolist()

        self.context_cat_cols = context_cols or []
        if self.context_cat_cols:
            append_dt_extracts(self.interactions_df, self.context_cat_cols)
            if not existing_cats_d:
                for col in self.context_cat_cols:
                    self.cats_d[col] = self.interactions_df[col]\
                        .cat.categories.tolist()

        self.make_feat_codes()
        self.process_num()

        # Alias Attributes
        self.user_col = self.cols[FGroup.USER]
        self.item_col = self.cols[FGroup.ITEM]
        self.user_feats_codes_df = self.feats_codes_df[FGroup.USER]
        self.item_feats_codes_df = self.feats_codes_df[FGroup.ITEM]
        self.context_feats_codes_df = self.feats_codes_df[FGroup.CONTEXT] \
            if FGroup.CONTEXT in self.feats_codes_df else None
        self.user_num_feats_df = self.num_feats_df[FGroup.USER]
        self.item_num_feats_df = self.num_feats_df[FGroup.ITEM]
        self.user_cat_cols = self.cat_cols[FGroup.USER]
        self.item_cat_cols = self.cat_cols[FGroup.ITEM]
        self.user_feats_d = self.feats_by_group[FGroup.USER]
        self.item_feats_d = self.feats_by_group[FGroup.ITEM]
        self.n_users = self.interactions_df[self.user_col].nunique()
        self.n_items = self.interactions_df[self.item_col].nunique()

    def export_data_encoding(self):
        return (self.cats_d,
                self.feats_codes_df[FGroup.USER],
                self.feats_codes_df[FGroup.ITEM],
                )

    def make_feat_codes(self):
        # Convert all categorical cols to corresponding codes
        for fgroup in [FGroup.USER, FGroup.ITEM]:
            self.feats_codes_df[fgroup] = \
                self.feats_by_group[fgroup][FType.CAT].copy()
            for col in self.feats_codes_df[fgroup].columns:
                self.feats_codes_df[fgroup][col] = \
                    self.feats_codes_df[fgroup][col].cat.codes

        if self.context_cat_cols:
            self.feats_codes_df[FGroup.CONTEXT] = \
                self.interactions_df[self.context_cat_cols].copy()
            for col in self.feats_codes_df[FGroup.CONTEXT].columns:
                self.feats_codes_df[FGroup.CONTEXT][col] = \
                    self.feats_codes_df[FGroup.CONTEXT][col].cat.codes

    def process_num(self):
        # Process numerical metadata
        # TODO: assuming numerical features aggregated into 1 table for now
        # ^ else, `self.user_feats_d[FType.NUM]: Iterable`

        self.num_meta = {}
        for fgroup in [FGroup.USER, FGroup.ITEM]:
            self.num_feats_df[fgroup] = \
                self.feats_by_group[fgroup][FType.NUM] \
                    if FType.NUM in self.feats_by_group[fgroup] else None

            # Gather metadata (size) of num feats
            if self.num_feats_df[fgroup] is not None:
                self.num_meta[fgroup] = self.num_feats_df[fgroup].shape[1]


def cast_cat(feats_d: Dict[FType, pd.DataFrame],
             existing_cats_d: Optional[Dict[str, Iterable]] = None,
             add_new_cats: Optional[bool] = False,
             ) -> Dict[FType, pd.DataFrame]:
    """Casts feature columns to categorical
    -- optionally applying existing categories

    Args:
        feats_d: Dictionary of feature dataframes
        existing_cats_d: Optional dictionary of existing categories.
            Note: these existing categories will be casted to the dtype of
            the column being casted.
        add_new_cats: if `True`, will append newly seen categories to
            book-keeping dictionary of categories (mutates inplace)

    Returns:
        Modified version of `feats_d`
    """

    for col in feats_d[FType.CAT].columns:
        if existing_cats_d and col in existing_cats_d:
            # Cast existing category to proper dtype (in-place)
            col_dtype = feats_d[FType.CAT][col].dtype
            existing_cats_d.update(
                {col: list(np.array(existing_cats_d[col]).astype(col_dtype))}
            )
            if add_new_cats:
                existing_cats_d[col] += list(
                    set(feats_d[FType.CAT][col].unique()) -
                    set(existing_cats_d[col]))
            existing_cats = existing_cats_d[col]
        else:
            existing_cats = None
        feats_d[FType.CAT][col] = feats_d[FType.CAT][col].astype(
            CategoricalDtype(existing_cats))
    return feats_d


def load_simple(
        interactions_src: InteractionsSource,
        features_srcs: FeatureSourceDictType,
        specific_feature: Dict[FGroup, bool],
        existing_cats_d: Optional[Dict[str, List[Any]]]=None,
        add_new_cats: Optional[bool] = False,
) -> Tuple[pd.DataFrame, Dict[FGroup, Dict[FType, pd.DataFrame]]]:
    """Stand-in loader mostly for local testing

    Args:
        interactions_src: Interactions data source
        features_srcs: Feature data source(s)
            keyed by group (user, item)
            if `None`, the primary id will be the sole feature
        specific_feature: If `True`, includes a primary id as a feature for
            that group (user, item)
        existing_cats_d: Optional dictionary of existing categories

    Returns:
        Tuple of preprocessed interactions, user features, and item_features
    """

    interactions_df = interactions_src.load().data

    cols = {
        FGroup.USER: interactions_src.user_col,
        FGroup.ITEM: interactions_src.item_col,
    }
    feats_by_group = {}

    for fgroup in [FGroup.USER, FGroup.ITEM]:
        src_l = features_srcs[fgroup]
        col = cols[fgroup]
        feats = {}

        if src_l:
            feats.update(load_many_srcs(src_l))
        if not src_l or not any([src.feature_type == FType.CAT
                                 for src in src_l]):
            feats[FType.CAT] = pd.DataFrame(
                index=interactions_df[col].drop_duplicates())
            specific_feature[fgroup] = True

        # TODO: Extreme simplifying assumption (should be handled better):
        #     This gets rid of missing side features
        feats[FType.CAT].dropna(axis=0, inplace=True)

        feats_by_group[fgroup] = feats

    # TODO: Another simplifying assumption:
    interactions_df, user_feats_d, item_feats_d, = simplifying_assumption(
        interactions_df,
        feats_by_group[FGroup.USER], feats_by_group[FGroup.ITEM],
        cols[FGroup.USER], cols[FGroup.ITEM],
    )

    for fgroup in [FGroup.USER, FGroup.ITEM]:
        feats = feats_by_group[fgroup]
        col = cols[fgroup]
        # index alignment for numerical features
        if FType.NUM in feats:
            feats[FType.NUM] = feats[FType.NUM] \
                .loc[interactions_df[col].unique()]

        # Use in the index as a feature (user and item specific eye feature)
        if specific_feature[fgroup]:
            feats[FType.CAT][feats[FType.CAT].index.name] = \
                feats[FType.CAT].index

        # Cast categorical
        feats = cast_cat(feats, existing_cats_d, add_new_cats)

        existing_fgroup_cats = feats[FType.CAT][col].cat.categories \
            if col in feats[FType.CAT] else None
        interactions_df[col] = interactions_df[col].astype(
            CategoricalDtype(existing_fgroup_cats))

        feats_by_group[fgroup] = feats

    log_shape_or_npartitions(interactions_df, 'interactions_df')
    log_shape_or_npartitions(
        feats_by_group[FGroup.USER][FType.CAT], 'user features')
    log_shape_or_npartitions(
        feats_by_group[FGroup.ITEM][FType.CAT], 'item_features')

    return interactions_df, feats_by_group


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
        CategoricalDtype(users_filt))
    interactions_df[item_col] = interactions_df[item_col].astype(
        CategoricalDtype(items_filt))

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
