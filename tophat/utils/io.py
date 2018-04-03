import fastavro as avro
from pathlib import Path
from lib_cerebro_py.custom_io import *
from typing import Dict, Union, List, Any, Optional


def write_vocab(vocab_dir: Union[str, Path],
                cats_d: Dict[str, List[Any]],
                ):
    """Writes a dictionary of categories to vocab files
    Each line of the file will contain 1 word of the vocabulary
    """
    vocab_dir = Path(vocab_dir)
    for k, v in cats_d.items():
        with open(vocab_dir / f'{k}.vocab', 'w') as f:
            f.write('\n'.join(map(str, v)) + '\n')


def load_vocab(vocab_dir: Union[str, Path],
               pattern: Optional[str] = '*',
               ) -> Dict[str, List[Any]]:
    """Loads a dictionary of categories from a directory of vocab files
    
    Args:
        vocab_dir: directory containing vocab files
        pattern: glob pattern for finding vocab files. 
            Note: vocab files created by `write_vocab` will have `.vocab` ext

    Returns: dictionary of vocab lists

    """
    vocab_dir = Path(vocab_dir)
    # WARNING: this reads the vocab as str type (could have been Any type)
    cats_d = {}
    for vocab_path in vocab_dir.glob(pattern):
        with open(vocab_path, 'r') as f:
            v = f.read().splitlines()
            cats_d[vocab_path.stem] = v
    return cats_d


# This should be moved to libcerepy
def import_pickle(path, compression='infer'):
    open_fun = open_fun_via_path(path, compression=compression)
    decompress_fun = decompress_fun_via_path(path, compression=compression)

    if isinstance(path, str):
        if AwsS3Uri.is_valid(path):
            with AwsS3Object(path).download_file_object() as data:
                return pickle.loads(decompress_fun(data.read()))
        else:
            return pickle.load(open_fun(path, 'rb'))
    else:
        return pickle.loads(decompress_fun(path))


def read_avro(path_or_buf) -> pd.DataFrame:
    if isinstance(path_or_buf, str):
        buf = open(path_or_buf, 'rb')
    else:
        buf = path_or_buf

    reader = avro.reader(buf)
    df = pd.DataFrame(list(reader))
    try:
        buf.close()
    except AttributeError:
        logger.info('Stream has no attribute close')

    return df


def load_factors(dir_factors: str) -> Dict[str, pd.DataFrame]:
    """
    
    Args:
        dir_factors: directory of various factors
            Ex) 's3://cerebro-recengine-exports/tophat/factors-prod'

    Returns:
        factors_d: dictionary of factor data frames keyed by factor group
        
    """

    factor_names = {x.uri.split('/')[-2]
                    for x in AwsS3Object(dir_factors).list_objects()}

    factors_d = {
        factor_name: dd_from_parts(
            os.path.join(dir_factors, factor_name),
            file_format='avro', limit_dates=False
        ).set_index('id').compute()
        for factor_name in factor_names
    }

    return factors_d


def cerebro_xn_load(path,
                    date_lookforward=None,
                    days_lookback=365,
                    assign_dates=False,
                    ):
    """

    Args:
        path: path for data
        assign_dates: If True, and loading from date-partitioned
                directories, assign the date information to a column
        days_lookback: Max number of days to look back from today
        date_lookforward: Furthest (most recent) date to consider

    Returns: dataframe of interactions

    """
    if os.path.splitext(path)[-1]:
        # single file -- can't selectively read partitions by date
        interactions_df = try_load(
            path,
            limit_dates=False)
    else:
        interactions_df = try_load(
            path,
            limit_dates=True,
            days_lookback=days_lookback,
            date_lookforward=date_lookforward,
            assign_dates=assign_dates,
        )

    return interactions_df


def cerebro_feature_load(path):
    feat_df = try_load(path, limit_dates=False)
    return feat_df
