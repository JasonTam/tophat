import argparse
from time import time

import numpy as np
import os
import tensorflow as tf
from lib_cerebro_py.log import logger

from tophat.data import TrainDataLoader
from tophat.embedding import EmbeddingMap, EmbeddingProjector
from tophat.evaluation import Validator
from tophat.nets import BilinearNetWithNum
from tophat.sampling import pair_sampler
from tophat.tasks.factorization import FactorizationTask
from tophat.utils.config_parser import Config

env = 'local'

EMB_DIM = 16
n_steps = 50000+1
log_every = 100
eval_every = 1000
save_every = 50000


def run(config):
    batch_size = config.get('batch_size')
    train_data_loader = TrainDataLoader(
        interactions_train=config.get('interactions_train'),
        user_features=config.get('user_features'),
        item_features=config.get('item_features'),
        user_specific_feature=config.get('user_specific_feature'),
        item_specific_feature=config.get('item_specific_feature'),
        context_cols=config.get('context_cols'),
        batch_size=config.get('batch_size'),
    )
    validator = Validator(config, train_data_loader,
                          limit_items=40000, n_users_eval=500,
                          include_cold=True, cold_only=False, n_xns_as_cold=5)

    # Ops and feature map
    logger.info('Building graph ...')
    embedding_map = EmbeddingMap(
        cats_d=train_data_loader.cats_d,
        user_cat_cols=train_data_loader.user_cat_cols,
        item_cat_cols=train_data_loader.item_cat_cols,
        context_cat_cols=train_data_loader.context_cat_cols,
        embedding_dim=EMB_DIM,
        zero_init_rows=validator.zero_init_rows,
        l2_bias=0., l2_emb=1e-4,
        vis_emb_user_col=train_data_loader.user_col,
    )

    model = FactorizationTask(
        net=BilinearNetWithNum(
            embedding_map=embedding_map,
            num_meta=train_data_loader.num_meta,
            l2_vis=0.,
            ruin=True),
        batch_size=batch_size,
    )

    loss = model.get_loss()
    train_op = model.training(loss)
    validator.make_ops(model)

    # Sample Generator
    logger.info('Setting up local sampler ...')
    sampler = pair_sampler.PairSampler.from_data_loader(
        train_data_loader,
        model.input_pair_d,
        batch_size=batch_size,
        method='uniform',
        uniform_users=True,
    )
    feed_dict_gen = iter(sampler)

    # Fit Loop
    n_interactions = len(train_data_loader.interactions_df)
    logger.info(f'Approx n_epochs: {(n_steps * batch_size) / n_interactions}')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
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

            if (step % log_every == 0) and step > 0:
                summary_str = sess.run(summary, feed_dict=feed_pair_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                toc = time() - tic
                tic = time()
                logger.info('(%.3f sec) \t Step %d: \t (train)loss = %.8f ' % (
                    toc, step, loss_val))

                if step % eval_every == 0:
                    validator.run_val(sess, summary_writer, step)

                if (step % save_every == 0) and step > 0:
                    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), step)

        embedding_projector.viz()

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Fit and evaluate on amazon womens small dataset')
    parser.add_argument('method', help='bpr or vbpr',
                        default='bpr', nargs='?',
                        choices=['bpr', 'vbpr'])
    args = parser.parse_args()
    config = Config(f'config/config_amzn_{args.method}.py')
    SEED = config.get('seed')
    np.random.seed(SEED)
    LOG_DIR = config.get('log_dir')
    tf.gfile.MkDir(LOG_DIR)
    run(config)
