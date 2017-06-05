import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from time import time, strftime, gmtime

import os

from tiefrex.core import EmbeddingMap, FactModel, fwd_dict_via_cats, pair_dict_via_cols
from tiefrex.config.main_cfg import main_cfg
from tiefrex.metadata_proc import write_metadata_emb
from tiefrex import naive_sampler
from tiefrex import data
from lib_cerebro_py.log import logger


path_interactions = main_cfg['local']['paths_input']['target_interactions']
path_item_features = main_cfg['local']['paths_input']['features']['item']['dim_products']['path']

activity_col = main_cfg['local']['activity_col']
activity_filter = main_cfg['local']['filter_activity_set']

user_col = main_cfg['local']['user_col']
item_col = main_cfg['local']['item_col']

SEED = main_cfg['seed']
np.random.seed(SEED)

EMB_DIM = 16
batch_size = 1024
n_steps = 2000
log_every = 100
LOG_DIR = f'/tmp/tensorboard-logs/{strftime("%Y-%m-%d-T%H%M%S", gmtime())}'
tf.gfile.MkDir(LOG_DIR)


def run():
    # LOAD THINGS
    interactions_df, user_feats_df, item_feats_df = data.load_simple(
        path_interactions, None, path_item_features,
        user_col, item_col,
        activity_col, activity_filter
    )
    user_feat_cols = user_feats_df.columns.tolist()
    item_feat_cols = item_feats_df.columns.tolist()

    cats_d = {
        **{feat_name: user_feats_df[feat_name].cat.categories.tolist()
           for feat_name in user_feat_cols},
        **{feat_name: item_feats_df[feat_name].cat.categories.tolist()
           for feat_name in item_feat_cols},
    }

    # INIT MODEL

    # Make Placeholders according to our cats
    with tf.name_scope('placeholders'):
        input_pair_d = pair_dict_via_cols(
            user_feat_cols, item_feat_cols, batch_size)

    # Ops and feature map
    embedding_map = EmbeddingMap(cats_d, user_feat_cols,
                                 item_feat_cols, embedding_dim=EMB_DIM)
    model = FactModel(embedding_map=embedding_map)
    loss = model.get_loss(input_pair_d)
    train_op = model.training(loss)

    # Convert everything to categorical codes
    user_feats_codes_df = user_feats_df.copy()
    for col in user_feats_codes_df.columns:
        user_feats_codes_df[col] = user_feats_codes_df[col].cat.codes
    item_feats_codes_df = item_feats_df.copy()
    for col in item_feats_codes_df.columns:
        item_feats_codes_df[col] = item_feats_codes_df[col].cat.codes

    # Assume we sample only out of items in our catalog
    item_ids = cats_d[item_col]

    # Sample Generator
    shuffle_inds = np.arange(len(interactions_df))
    feed_dict_gen = naive_sampler.feed_dicter(
        shuffle_inds, batch_size,
        interactions_df,
        user_col, item_col,
        user_feats_codes_df, item_feats_codes_df,
        item_ids, input_pair_d)

    # Fit Loop
    n_interactions = len(interactions_df)
    logger.info(f'Approx n_epochs: {(n_steps * batch_size) / n_interactions}')

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(
            LOG_DIR, graph=tf.get_default_graph())

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

        # serialize meta data for embedding viz
        feat_to_metapath = write_metadata_emb(
            cats_d, main_cfg['local']['paths_input']['names'], LOG_DIR)

        config = projector.ProjectorConfig()
        emb_proj_obj_d = {}
        for feat_name, emb in embedding_map.embeddings_d.items():
            if feat_name in feat_to_metapath:
                emb_proj_obj_d[feat_name] = config.embeddings.add()
                emb_proj_obj_d[feat_name].tensor_name = emb.name
                emb_proj_obj_d[feat_name].metadata_path = feat_to_metapath[feat_name]

        # After the last step, lets save some embedding to viz later
        projector.visualize_embeddings(summary_writer, config)
        saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), step)

    return True


if __name__ == '__main__':
    run()
