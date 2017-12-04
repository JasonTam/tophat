from time import time

import numpy as np
import os
import tensorflow as tf
from lib_cerebro_py.log import logger

from tophat import naive_sampler
from tophat.config_parser import Config
from tophat.core import FactModel
from tophat.data import TrainDataLoader
from tophat.evaluation import Validator
from tophat.embedding import EmbeddingMap, EmbeddingProjector
from tophat.nets import BilinearNet

config = Config('config/config_context.py')

env = 'local'

SEED = config.get('seed')
np.random.seed(SEED)

EMB_DIM = 16
batch_size = config.get('batch_size')
n_steps = 50000+1
log_every = 100
eval_every = 1
save_every = 50000
LOG_DIR = config.get('log_dir')
tf.gfile.MkDir(LOG_DIR)


def run():
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
                          limit_items=-1, n_users_eval=500,
                          include_cold=False, cold_only=False)

    # Ops and feature map
    logger.info('Building graph ...')
    embedding_map = EmbeddingMap(train_data_loader, embedding_dim=EMB_DIM,
                                 zero_init_rows=validator.zero_init_rows,
                                 vis_specific_embs=False,
                                 feature_weights_d=config.get('feature_weights_d'),
                                 )

    model = FactModel(
        net=BilinearNet(
            embedding_map=embedding_map,
            interaction_type=config.get('interaction_type')
        )
    )
    # model = FactModel(net=BilinearNetWithNum(
    #     embedding_map=embedding_map,
    #     interaction_type=config.get('interaction_type'),
    #     num_meta=train_data_loader.num_meta),
    #
    # )
    # model = FactModel(net=BilinearNetWithNumFC(
    #     embedding_map=embedding_map,
    #     interaction_type=config.get('interaction_type'),
    #     num_meta=train_data_loader.num_meta)
    # )

    loss = model.get_loss()
    train_op = model.training(loss)
    validator.make_ops(model)

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
                    validator.run_val_context(sess, summary_writer, step)

                if (step % save_every == 0) and step > 0:
                    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), step)

        embedding_projector.viz()

    return True


if __name__ == '__main__':
    run()
