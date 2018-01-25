import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import os
from collections import defaultdict
import itertools as it
from time import time
import argparse

from joblib import Parallel, delayed, cpu_count
from functools import partial

import pprint
from tophat.constants import FType
from tophat.data import FeatureSource, load_many_srcs
from tophat.config_parser import Config
from tophat.utils.io import load_factors, import_pickle
from tophat.export_job import write_avro
from lib_cerebro_py.log import logger, log_shape_or_npartitions
from lib_cerebro_py.custom_io import dd_from_parts
from typing import Dict, List


class ColdRepresentationExportJob(object):
    """ Exports representations for cold records with features
    Valid records must have content-based features and existing factors 
    for all respective feature
    
    Args:
        paths_cold_d: paths for cold records to process
        user_features: source(s) for features for cold users
        item_features: source(s) for features for cold items
        feature_weights_d: dictionary of feature weights
            (should be the same as in training)
        dir_factors: directory of factors
        repr_export_path: path to export cold representations
        n_partitions: number of partitions in export
        n_jobs: number of workers for exporting
    """

    def __init__(self,
                 paths_cold_d: Dict[str, str],
                 user_features: List[FeatureSource],
                 item_features: List[FeatureSource],
                 path_cats: str,
                 feature_weights_d: Dict[str, float],
                 dir_factors: str,
                 repr_export_path: str,
                 n_partitions: int =8,
                 n_jobs: int =1,
                 ):

        self.paths_cold = paths_cold_d
        self.user_features = user_features
        self.item_features = item_features
        self.path_cats = path_cats
        self.feature_weights_d = defaultdict(lambda: 1., feature_weights_d)
        self.dir_factors = dir_factors
        self.repr_export_path = repr_export_path
        self.n_partitions = n_partitions
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    def run(self) -> None:

        # Load Cats
        cats_d = import_pickle(self.path_cats)

        # Load Factors
        factors_d = load_factors(self.dir_factors)
        emb_dim = len(list(factors_d.values())[0].factors.iloc[0])

        # Load Features
        # Note: Only categorical features supported for cold-start imputation

        for user_or_item, feature_src in [('user', self.user_features),
                                          ('item', self.item_features)]:

            features_df = load_many_srcs(feature_src)[FType.CAT]
            if not isinstance(features_df, pd.DataFrame):
                logger.info(f'There are no {user_or_item} content features')
                continue
            ind_col = features_df.index.name

            # Mapping to categorical codes
            # Note: cat to code is faster than mapping with dict
            codes_df = pd.DataFrame(index=features_df.index)
            for col in codes_df.columns:
                if col in cats_d:
                    # Modify in-place
                    codes_df[col] = codes_df[col].astype(
                        CategoricalDtype(categories=cats_d[col]))

            # Load and process records in paths of cold items
            ind_dtype = features_df.index.dtype

            warm_records = set(
                factors_d[ind_col].index.astype(ind_dtype))

            cold_records = set(it.chain(*(set(
                dd_from_parts(
                    path_cold, file_format='msg', limit_dates=False
                ).compute()[ind_col].astype(ind_dtype))
                for path_cold in self.paths_cold[user_or_item]))
                               ) - warm_records
            n_cold_total = len(cold_records)
            cold_records = list(
                cold_records.intersection(set(features_df.index)))
            n_cold_processable = len(cold_records)
            logger.info(f'{n_cold_processable} processable cold records'
                        f' out of {n_cold_total} provided records')

            if not len(cold_records):
                logger.info(f'There are no {user_or_item} records to process')
                continue

            # Weighing and fillna for feature-factors of cold records
            cold_feats_df = features_df.loc[cold_records]
            cold_bias_arr = np.zeros(len(cold_feats_df), float)
            cold_factors_arr = np.zeros((len(cold_feats_df), emb_dim), float)

            for col in cold_feats_df.columns:
                w = self.feature_weights_d[col]
                cold_component = factors_d[col].reindex(
                    cold_feats_df[col].astype(str))
                bias_component = w * cold_component.bias.fillna(0.).values
                factor_component = w * np.vstack(cold_component.factors.apply(
                    # Ghetto fillna
                    lambda x: x if isinstance(x, list) else [0.]*emb_dim))

                cold_bias_arr += bias_component
                cold_factors_arr += factor_component

            cold_bias_df = pd.DataFrame(
                cold_bias_arr,
                index=cold_feats_df.index,
                columns=['bias'])
            cold_embs_df = pd.DataFrame(
                cold_factors_arr,
                index=cold_feats_df.index,
                columns=[f'factor_{ii}' for ii in range(emb_dim)])
            reprs_df = pd.concat([cold_embs_df, cold_bias_df], axis=1)

            if n_cold_processable > 1e4:
                n_partitions = self.n_partitions
            else:
                n_partitions = 1

            # TODO: idk why parallel is slower and less stable
            # Parallel(n_jobs=self.n_jobs) \
            #     (delayed(partial(write_avro,
            #                      dir_export=self.repr_export_path,
            #                      col_name=ind_col,
            #                      n_partitions=n_partitions,
            #                      factors_df=reprs_df,
            #                      part_name='cold',
            #                      ))(partition)
            #      for partition in range(n_partitions))
            for partition in range(n_partitions):
                write_avro(
                    dir_export=self.repr_export_path,
                    col_name=ind_col,
                    n_partitions=n_partitions,
                    factors_df=reprs_df,
                    part_name='cold',
                    partition=partition,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export cold user / item representations from model')
    parser.add_argument('environment', help='Run environment',
                        default='prod', nargs='?',
                        choices=['prod'])
    args = parser.parse_args()

    config = Config(f'config/cold_config_{args.environment}.py')

    job = ColdRepresentationExportJob(
        paths_cold_d=config.get('paths_cold_d'),
        user_features=config.get('user_features'),
        item_features=config.get('item_features'),
        path_cats=config.get('path_cats'),
        feature_weights_d=config.get('feature_weights_d'),
        dir_factors=config.get('dir_factors'),
        repr_export_path=config.get('repr_export_path'),
        n_partitions=config.get('n_partitions'),
        n_jobs=config.get('n_jobs') or 1,
    )
    job.run()
