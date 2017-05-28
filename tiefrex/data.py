import pandas as pd

from lib_cerebro_py import custom_io
from lib_cerebro_py.log import logger, log_shape_or_npartitions


def load_simple(
        path_interactions,
        path_user_features, path_item_features,
        user_col, item_col,
        activity_col, activity_filter,):
    """Stand-in loader mostly for local testing"""

    interactions_df = custom_io.try_load(path_interactions)
    interactions_df = custom_io.filter_col_isin(interactions_df, activity_col, activity_filter).compute()

    if path_user_features:
        user_feats_df = custom_io \
            .try_load(path_item_features, limit_dates=False) \
            .set_index(item_col)
    else:
        user_feats_df = pd.DataFrame(index=interactions_df[user_col].drop_duplicates())
    if path_item_features:
        item_feats_df = custom_io \
            .try_load(path_item_features, limit_dates=False) \
            .set_index(item_col)
    else:
        item_feats_df = pd.DataFrame(index=interactions_df[item_col].drop_duplicates())

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
