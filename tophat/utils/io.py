import os
import pandas as pd
import pickle
from lib_cerebro_py.custom_io import *
from typing import Dict


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