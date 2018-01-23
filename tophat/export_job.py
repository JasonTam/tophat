# some repeat between this and lightfm-shenanigans repo

import argparse
import fastavro as avro
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tophat import schemas
from tophat.config_parser import Config
from joblib import Parallel, delayed, cpu_count
from functools import partial
from lib_cerebro_py.aws.aws_s3_uri import AwsS3Uri
from lib_cerebro_py.aws.aws_s3_object import AwsS3Object
from lib_cerebro_py.log import logger
from tophat.convenience import import_pickle
from typing import Iterator, Dict, Any, List
import pprint


def rec_generator(df: pd.DataFrame,
                  partition: int,
                  n_partitions: int =1) -> Iterator[Dict[str, Any]]:
    """
    partition : the worker number (determines which stratification to process)
    n_partitions : total number of partitions
    """
    partition_inds = range(partition, len(df), n_partitions)
    part_df = df.iloc[partition_inds]
    factors_l = part_df.filter(like='factor_').values.tolist()
    part_df.index.name = 'id'
    part_df.reset_index(inplace=True)
    for rec_i in range(len(partition_inds)):
        # did not use .to_dict('records') cuz of minor postproc on copy view
        record = {
            'id': str(part_df['id'].iloc[rec_i]),
            'factors': list(map(float, factors_l[rec_i])),
            'bias': float(part_df['bias'].iloc[rec_i]),
        }
        yield record


def write_avro(partition: int,
               n_partitions: int,
               dir_export: str, col_name: str,
               factors_df: pd.DataFrame,
               part_name: str='part',
               ) -> str:
    """
    col_name : identifies what the feature representation is based on
        (gender, brand_id, ...)
    """
    rec_gen = rec_generator(factors_df, partition, n_partitions)

    part_name = os.path.join(col_name, f'{part_name}.{partition}.avro')

    if AwsS3Uri.is_valid(dir_export):
        path_upload = os.path.join(dir_export, part_name)
        path_out = os.path.join('/tmp', part_name)
    else:
        path_out = os.path.join(dir_export, part_name)

    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))

    with open(path_out, 'wb') as f_out:
        avro.writer(f_out, schemas.factors_avro, rec_gen, codec='snappy')

    if AwsS3Uri.is_valid(dir_export):
        AwsS3Object(path_upload).upload_file(path_out)

    return path_out


def export_factors_df(df: pd.DataFrame, path: str) -> None:
    # TODO: pandas issue #15487 cannot `pd.read_msgpack` with
    #   `CategoricalIndex` until version 0.20.0
    # (use p/hdf for now)
    if AwsS3Uri.is_valid(path):
        # df.to_hdf('/tmp/to_upload', key='whatever', mode='w')
        df.to_pickle('/tmp/to_upload')
        AwsS3Object(path).upload_file('/tmp/to_upload')
    else:
        # df.to_hdf(path, key='whatever', mode='w')
        df.to_pickle(path)
    logger.info('Factors exported to %s' % path)


class FactorExportJob(object):
    def __init__(self,
                 path_cats: str,
                 path_meta: str,
                 path_ckpt: str,
                 dir_export: str,
                 export_cols: List[str]=None,
                 n_partitions: int =8,
                 n_jobs: int =1,
                 export_pickle: bool =False,
                 ):
        """
        :param dir_export: directory to export factors
        :param export_cols: if `None`, export all cols in graph meta
            else, a list of feature-group names to export
            to export
        :param n_partitions: number of partitions per export
        :param n_jobs: number of processing jobs
        :param export_pickle: if `True` export pickled version as well
        """
        self.path_cats = path_cats
        self.path_meta = path_meta
        self.path_ckpt = path_ckpt
        self.dir_export = dir_export

        self.export_cols = export_cols
        self.n_partitions = n_partitions
        self.n_jobs = n_jobs
        self.export_pickle = export_pickle

        self.cats_d = None
        self.saver = None

    def extract_via_tf(self):
        self.cats_d = import_pickle(self.path_cats)

        self.saver = tf.train.import_meta_graph(self.path_meta)

        # Extract Relevant Variables
        emb_vars = [v for v in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='embeddings',
        ) if '/Adam' not in v.name]  # Don't get the adam state vars
        bias_vars = [v for v in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='biases',
        ) if '/Adam' not in v.name]  # Don't get the adam state vars

        # Evaluate Vars via checkpoint
        sess_config = tf.ConfigProto(
            device_count={'GPU': 0}  # run on CPU
        )
        with tf.Session(config=sess_config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.saver.restore(sess, self.path_ckpt)
            embs_eval = sess.run(emb_vars)
            bias_eval = sess.run(bias_vars)

        # TODO: consider removing '_biases' and '_embs' as it is redundant
        #   (so we don't have to do that last split on '_')
        var_emb_names = [v.name.split('/')[-1].split(':')[0].rsplit('_', 1)[0]
                         for v in emb_vars]
        var_bias_names = [v.name.split('/')[-1].split(':')[0].rsplit('_', 1)[0]
                          for v in bias_vars]

        # In case they are mis-aligned (should not be the case)
        emb_eval_d = dict(zip(var_emb_names, embs_eval))
        bias_eval_d = dict(zip(var_bias_names, bias_eval))

        return emb_eval_d, bias_eval_d

    def actually_export(self, df_d):
        # Finally, export
        export_cols = self.export_cols or list(df_d.keys())
        for col in export_cols:
            factors_df = df_d[col]

            if len(factors_df) / 1e4 > 1:
                feature_n_partitions = self.n_partitions
            else:
                feature_n_partitions = 1
            Parallel(n_jobs=self.n_jobs) \
                (delayed(partial(write_avro,
                                 dir_export=self.dir_export,
                                 col_name=col,
                                 n_partitions=feature_n_partitions,
                                 factors_df=factors_df,
                                 ))(partition)
                 for partition in range(feature_n_partitions))

    def make_df(self, emb_eval_d, bias_eval_d):
        # Make DataFrames
        df_d = {}
        for var_name in emb_eval_d.keys():
            index = pd.Index(self.cats_d[var_name], name=var_name)
            emb_data = emb_eval_d[var_name]
            bias_data = bias_eval_d[var_name]

            cols = [f'factor_{i}' for i in range(emb_data.shape[1])] + ['bias']

            # Note: Lightfm export style
            #   (can consider doing something more direct)
            df = pd.DataFrame(np.hstack([emb_data, bias_data[:, None]]),
                              index=index, columns=cols)
            df_d[var_name] = df

        return df_d

    def run(self) -> None:
        emb_eval_d, bias_eval_d = self.extract_via_tf()
        df_d = self.make_df(emb_eval_d, bias_eval_d)
        self.actually_export(df_d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export factors from a checkpointed tophat model')
    parser.add_argument('environment', help='Run environment',
                        default='prod', nargs='?',
                        choices=['integ', 'prod'])

    args = parser.parse_args()
    logger.info(pprint.pformat(args))
    config = Config(f'config/export_config_{args.environment}.py')

    job = FactorExportJob(
        config.get('path_cats'),
        config.get('path_meta'),
        config.get('path_ckpt'),
        config.get('dir_export'),
    )
    job.run()





