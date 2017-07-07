import numpy as np
import tensorflow as tf
from time import time

import os

from tiefrex.core import EmbeddingMap, EmbeddingProjector, FactModel
from tiefrex.config.main_cfg import main_cfg
from tiefrex.config.eval_cfg import eval_cfg
from tiefrex import naive_sampler
from tiefrex.data import TrainDataLoader
from tiefrex.evaluation import Validator
from tiefrex.config_parser import Config
from lib_cerebro_py.log import logger
from tiefrex.constants import FType

config = Config('tiefrex/config/config.py')

env = 'local'

SEED = config.get('seed')
np.random.seed(SEED)

EMB_DIM = 16
batch_size = config.get('batch_size')
n_steps = 2000
log_every = 100
eval_every = 500
LOG_DIR = config.get('log_dir')
tf.gfile.MkDir(LOG_DIR)


def run():
    train_data_loader = TrainDataLoader(config)
    cats_d, user_cat_codes_df, item_cat_codes_df = train_data_loader.export_data_encoding()
    # validator = Validator(
    #     config, cats_d,
    #     user_cat_codes_df, item_cat_codes_df,
    #     train_data_loader.user_num_feats_df,
    #     train_data_loader.item_num_feats_df,
    #     )
    validator = Validator(config, train_data_loader)

    # Ops and feature map
    logger.info('Building graph ...')
    embedding_map = EmbeddingMap(train_data_loader, embedding_dim=EMB_DIM)
    model = FactModel(embedding_map=embedding_map,
                      num_meta=train_data_loader.num_meta)
    loss = model.get_loss()
    train_op = model.training(loss)
    validator.ops(model)

    # Sample Generator
    logger.info('Setting up local sampler ...')
    sampler = naive_sampler.PairSampler(
        train_data_loader,
        model.input_pair_d,
        batch_size,
        method='uniform',
    )
    feed_dict_gen = iter(sampler)

    # Fit Loop
    n_interactions = len(train_data_loader.interactions_df)
    logger.info(f'Approx n_epochs: {(n_steps * batch_size) / n_interactions}')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(
            LOG_DIR, graph=tf.get_default_graph())
        embedding_projector = EmbeddingProjector(embedding_map, summary_writer, config)

        tic = time()
        for step in range(n_steps):
            feed_pair_dict = next(feed_dict_gen)
            _, loss_val = sess.run([train_op, loss], feed_dict=feed_pair_dict)

            if step % log_every == 0:
                summary_str = sess.run(summary, feed_dict=feed_pair_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                toc = time() - tic
                tic = time()
                logger.info('(%.3f sec) \t Step %d: \t (train)loss = %.8f ' % (
                    toc, step, loss_val))

                if step % eval_every == 0:
                    validator.run_val(sess, summary_writer, step)

        embedding_projector.viz()

        saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), step)

    return True


if __name__ == '__main__':
    run()
