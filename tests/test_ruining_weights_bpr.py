import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from time import time

import os

from tiefrex.core import EmbeddingProjector, FactModel
from tiefrex.nets import EmbeddingMap, BilinearNetWithNum

from tiefrex.data import TrainDataLoader
from tiefrex.evaluation import Validator
from tiefrex.config_parser import Config
from lib_cerebro_py.log import logger

config = Config('tiefrex/config/config_amzn_bpr.py')

env = 'local'

SEED = config.get('seed')
np.random.seed(SEED)

EMB_DIM = 16
batch_size = config.get('batch_size')
LOG_DIR = config.get('log_dir')
tf.gfile.MkDir(LOG_DIR)


def run():
    train_data_loader = TrainDataLoader(config)
    # will modify `train_data_loader` in-place
    validator = Validator(config, train_data_loader,
                          limit_items=-1, n_users_eval=500,
                          include_cold=True, cold_only=False)

    # Ops and feature map
    logger.info('Building graph ...')
    embedding_map = EmbeddingMap(train_data_loader, embedding_dim=EMB_DIM,
                                 zero_init_rows=validator.zero_init_rows,
                                 )

    model = FactModel(net=BilinearNetWithNum(
        embedding_map=embedding_map,
        num_meta=train_data_loader.num_meta)
    )

    # ------------------

    local_data_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../data/amazon/')

    path_gamma_user = os.path.join(local_data_dir, 'ruining_weights/BPR/gamma_user_df.msg')
    path_gamma_item = os.path.join(local_data_dir, 'ruining_weights/BPR/gamma_item_df.msg')
    path_beta_item = os.path.join(local_data_dir, 'ruining_weights/BPR/beta_item_df.msg')

    gamma_user_df = pd.read_msgpack(path_gamma_user)
    gamma_item_df = pd.read_msgpack(path_gamma_item)
    beta_item_df = pd.read_msgpack(path_beta_item)

    user_index = model.net.embedding_map.cats_d['reviewerID']
    item_index = model.net.embedding_map.cats_d['asin']

    gamma_user_w = gamma_user_df.loc[user_index].fillna(0.).values.astype('float32')
    gamma_item_w = gamma_item_df.loc[item_index].fillna(0.).values.astype('float32')
    beta_item_w = beta_item_df.loc[item_index].fillna(0.).values.astype('float32')

    # Remake the embeddings with predefined weights
    model.net.embedding_map.embeddings_d['reviewerID'] = tf.get_variable(
        name='reviewerID_embs',
        shape=gamma_user_w.shape,
        initializer=tf.constant_initializer(gamma_user_w),
    )
    model.net.embedding_map.embeddings_d['asin'] = tf.get_variable(
        name='asin_embs',
        shape=gamma_item_w.shape,
        initializer=tf.constant_initializer(gamma_item_w),
    )
    model.net.embedding_map.biases_d['asin'] = tf.get_variable(
        name='asin_biases',
        shape=beta_item_w.shape,
        initializer=tf.constant_initializer(beta_item_w),
    )

    # ---------------
    validator.make_ops(model)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(
            LOG_DIR, graph=tf.get_default_graph())
        embedding_projector = EmbeddingProjector(embedding_map, summary_writer, config)

        summary_writer.flush()
        tic = time()
        scores_d = validator.run_val(sess, summary_writer, 0)
        toc = time() - tic
        logger.info(f'Validation time: {toc}s')

        embedding_projector.viz()

        saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), 0)

    # Check AUC -- should be close to numpy evaluation
    # not exactly equal because we don't assert the same 500 users
    # see notebook for source of desired values
    # (500 users, limit -1 items, include cold, not cold only)
    np.testing.assert_almost_equal(np.mean(scores_d['auc']), 0.717719806398, decimal=2)
    np.testing.assert_almost_equal(np.std(scores_d['auc']), 0.2981886522559542, decimal=2)

    return True


if __name__ == '__main__':
    run()
