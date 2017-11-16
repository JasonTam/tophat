import numpy as np
import tensorflow as tf
from tophat.embedding import EmbeddingMap, lookup_wrapper
from tophat import ph_conversions
from tophat.constants import FType
from tophat.data import TrainDataLoader
from typing import Dict, Tuple, Iterator, Any

import os
import fastavro as avro
from joblib import Parallel, delayed, cpu_count
from lib_cerebro_py.aws.aws_s3_uri import AwsS3Uri
from lib_cerebro_py.aws.aws_s3_object import AwsS3Object
from lib_cerebro_py.log import logger
from tophat import schemas
from tqdm import tqdm
from time import time
from functools import partial


# TODO: COPY PASTA FROM lightfm-shenanigans (pls refactor)
def rec_generator(
        representations: np.array, id_map: Dict[int, Any],
        partition: int, n_partitions: int =1) -> Iterator[Dict[str, Any]]:
    """
    
    Args:
        representations: The representation vectors to export
        id_map: Mapping of index to id
        partition: Worker number (determines which stratification to process)
        n_partitions: Total number of partitions

    Yields:
        Dictionary record to export

    """
    for row_n in range(partition, representations.shape[0], n_partitions):
        row = representations[row_n]
        # The bias term is the last element of the representation array
        record = {
            'id': str(id_map[row_n]),
            'factors': list(map(float, row[:-1])),
            'bias': float(row[-1]),
        }
        yield record


def write_avro(partition: int,
               n_partitions: int,
               dir_export: str, col_name: str,
               id_map: Dict[int, Any], representations: np.array,
               ) -> str:
    """
    col_name : identifies what the {user/item} representation is based on
        (ops_user_id, ops_product_id)
    """
    rec_gen = rec_generator(representations, id_map, partition, n_partitions)

    part_name = os.path.join(col_name, 'part.{}.avro'.format(partition))

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


def calc_reprs_ops(embedding_map: EmbeddingMap,
                   input_group_d: Dict[str, tf.Tensor]
                   ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Calculate representations (lightfm style)

    Note: This only makes sense when the model is trained with
        `interaction_type='inter'` (this emulates lightfm formulation)
    """

    # Note: lookup_wrapper already applies the feature weight
    embeddings_d = lookup_wrapper(
        embedding_map.embeddings_d, input_group_d, cols=input_group_d.keys(),
        scope='repr_lookup', name_tmp='{}_emb',
        feature_weights_d=embedding_map.feature_weights_d,
    )

    biases_d = lookup_wrapper(
        embedding_map.biases_d, input_group_d, cols=input_group_d.keys(),
        scope='repr_lookup', name_tmp='{}_bias',
        feature_weights_d=embedding_map.feature_weights_d,
    )

    emb_repr_op = tf.reduce_sum(list(embeddings_d.values()), axis=0)
    bias_repr_op = tf.reduce_sum(list(biases_d.values()), axis=0)

    return emb_repr_op, bias_repr_op


class RepresentationExportJob(object):
    def __init__(self,
                 sess: tf.Session,  # TODO: tmp workaround
                 # path_import_model: str,  # TODO?
                 embedding_map: EmbeddingMap,
                 train_data_loader: TrainDataLoader,
                 dir_export: str,
                 n_partitions: int =8,
                 n_jobs: int =1,
                 batch_size: int=1024*8,
                 ):
        self.sess = sess
        # self.path_import_model = path_import_model  # TODO?
        self.train_data_loader = train_data_loader
        self.embedding_map = embedding_map
        self.cats_d = self.embedding_map.cats_d

        self.dir_export = dir_export
        self.n_partitions = n_partitions
        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def run(self) -> None:

        params_d = {
            'user': {
                'codes_df': self.train_data_loader.user_feats_codes_df,
                'col_name': self.train_data_loader.user_col,
                'cat_cols': self.train_data_loader.user_cat_cols,
            },
            'item': {
                'codes_df': self.train_data_loader.item_feats_codes_df,
                'col_name': self.train_data_loader.item_col,
                'cat_cols': self.train_data_loader.item_cat_cols,
            },
        }

        for params in params_d.values():
            tic = time()
            codes_df = params['codes_df']
            col_name = params['col_name']
            cat_cols = params['cat_cols']

            structs = {
                FType.CAT: cat_cols,
                FType.NUM: list(self.train_data_loader.num_meta.items()),
            }

            input_fwd_d = ph_conversions.fwd_dict_via_ftypemeta(
                structs, batch_size=None)

            repr_emb_op, repr_bias_op = calc_reprs_ops(
                self.embedding_map,
                input_fwd_d,
            )

            # This is a bit strange, the index of `codes_df` is not ordered
            #    the same as self.cats_d[col_name]
            # id_map = dict(enumerate(self.cats_d[col_name]))
            id_map = dict(enumerate(codes_df.index))

            chunks = []
            for ii in tqdm(range(0, len(codes_df), self.batch_size)):
                chunk_df = codes_df.iloc[ii:ii + self.batch_size]

                feed_d = chunk_df.to_dict(orient='list')

                feed_fwd_d = {input_fwd_d[f'{feat_name}']: data_in
                              for feat_name, data_in in feed_d.items()}

                chunk_emb_repr, chunk_bias_repr = self.sess.run(
                    [repr_emb_op, repr_bias_op], feed_dict=feed_fwd_d)

                chunks.append((chunk_emb_repr, chunk_bias_repr))

            emb_repr, bias_repr = map(np.concatenate, zip(*chunks))
            representations = np.hstack([emb_repr, bias_repr[:, None]])

            Parallel(n_jobs=self.n_jobs) \
                (delayed(partial(write_avro,
                                 dir_export=self.dir_export,
                                 col_name=col_name,
                                 n_partitions=self.n_partitions,
                                 id_map=id_map,
                                 representations=representations,
                                 ))(partition)
                 for partition in range(self.n_partitions))

            logger.info('...finished export of: {} representations in {}s'
                        .format(col_name, time() - tic))
