from tophat.utils.log import logger


def filter_col_isin(df, col, inclusion_set):
    if len(inclusion_set) and col in df.columns:
        logger.info('Filtering on {} in {}'.format(col, inclusion_set))
        return df.loc[df[col].isin(inclusion_set)]
    else:
        logger.info('Nothing to filter on')
        return df


def log_shape_or_npartitions(df, name: str ='') -> None:
    """
    df : dataframe to log shape or npartitions
    name : optional name of dataframe as extra info
    """
    if hasattr(df, 'compute'):  # if dask dataframe
        logger.info(f'{name} npartitions:\t({df.npartitions})')
    else:
        logger.info(f'{name} shape:\t(%d,%d)' % df.shape)
