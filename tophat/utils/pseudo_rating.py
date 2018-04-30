import numpy as np
import pandas as pd
from typing import Dict
from pandas.api.types import CategoricalDtype

pseudo_rating_weights = {
    b'purch': 0.8119,
    b'cart':  0.1500,
    b'list':  0.0444,
    b'click': 0.0312,
    b'visit': 0.0135,
}


def calc_pseudo_ratings(interactions_df: pd.DataFrame,
                        user_col: str, item_col: str,
                        weights_d: Dict[str, float] =pseudo_rating_weights,
                        counts_col: str ='counts',
                        weight_switch_col: str ='activity',
                        sublinear: bool =True,
                        reagg_counts: bool=False,
                        output_col: str ='pseudo_rating',
                        ) -> pd.DataFrame:
    """Synthesizes a pseudo-rating based on activity type and counts
    
    Args:
        interactions_df: DataFrame with counts of user, item, activity counts
        user_col: name of user column
        item_col: name of item column
        weights_d: dictionary of weights by activity type
        counts_col: name count column
        weight_switch_col: column name that maps to pseudo-rating weights
        sublinear: if True, apply log1p sublinear scaling to count
        reagg_counts: if True, re-aggregate interaction counts
        output_col: name of output pseudo-rating column

    Returns:
        Aggregated dataframe with pseudo-rating column

    """
    scaling_fn = np.log1p if sublinear else lambda x: x

    if reagg_counts:
        # Assure that counts are already aggregated
        df = interactions_df\
            .groupby([user_col, item_col, weight_switch_col])\
            .sum().reset_index()
    else:
        df = interactions_df

    df[output_col] = (
        df[weight_switch_col].map(weights_d).astype(np.float32) *
        df[counts_col].map(scaling_fn).astype(np.float32)
    )
    g = df.groupby([user_col, item_col])
    out_df = g[output_col].sum().reset_index()

    # Re-apply categories
    out_df[user_col] = out_df[user_col].astype(CategoricalDtype(
        categories=interactions_df[user_col].cat.categories))
    out_df[item_col] = out_df[item_col].astype(CategoricalDtype(
        categories=interactions_df[item_col].cat.categories))

    return out_df
