

def append_dt_extracts(df, cols, cats_d=None, date_col='date'):
    """ Appends dt derived columns to `df` in-place 
    """
    if not cols:
        return
    for attr in ['month', 'quarter']:
        if attr in cols:
            if cats_d and attr in cats_d:
                categories = cats_d[attr]
            else:
                categories = None
            df[attr] = getattr(df[date_col].dt, attr)\
                .astype('category', categories=categories)
