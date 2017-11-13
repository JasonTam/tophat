import tensorflow as tf
import numpy as np
import itertools as it
from tiefrex.constants import *
from tiefrex.ph_conversions import *
from tiefrex.metadata_proc import write_metadata_emb
from tiefrex.nets import *
from tensorflow.contrib.tensorboard.plugins import projector
from typing import List, Dict, Iterable


class EmbeddingProjector(object):
    def __init__(self, embedding_map, summary_writer, config):
        self.summary_writer = summary_writer
        feat_to_metapath = write_metadata_emb(
            embedding_map.data_loader.cats_d, config.get('names'), config.get('log_dir'))

        self.projection_config = projector.ProjectorConfig()
        emb_proj_obj_d = {}
        for feat_name, emb in embedding_map.embeddings_d.items():
            if feat_name in feat_to_metapath:
                emb_proj_obj_d[feat_name] = self.projection_config.embeddings.add()
                emb_proj_obj_d[feat_name].tensor_name = emb.name
                emb_proj_obj_d[feat_name].metadata_path = feat_to_metapath[feat_name]

    def viz(self):
        # After the last step, lets save some embedding to viz later
        projector.visualize_embeddings(self.summary_writer, self.projection_config)


class FactModel(object):

    def __init__(self,
                 net,
                 optimizer: tf.train.Optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
                 # optimizer: tf.train.Optimizer = tf.train.AdagradOptimizer(learning_rate=0.05),
                 # optimizer: tf.train.Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1),
                 seed=SEED,
                 ):
        """
        :param embedding_map: variables and metadata concerning categorical embeddings
        :param num_meta: metadata concerning numerical data
            `feature_name -> dimensionality of input`
        :param l2: l2 regularization scale (0 to disable)
        :param intra_field: flag to include intra-field interactions
        :param optimizer: tf optimizer
        """

        self.seed = seed
        self.net = net
        self.optimizer = optimizer
        self.input_pair_d: Dict[str, tf.Tensor] = None

        self.forward = self.net.forward

        # Make Placeholders according to our cats
        with tf.name_scope('placeholders'):
            self.input_pair_d = self.get_pair_dict(self.net.embedding_map.data_loader.batch_size)

    def get_fwd_dict(self, batch_size: int=None):
        return fwd_dict_via_cats(
            self.net.embedding_map.cats_d.keys(),
            batch_size)

    def get_pair_dict(self, batch_size: int=None):
        return pair_dict_via_ftypemeta(
            user_ftypemeta={
                FType.CAT: self.net.embedding_map.user_cat_cols,
                # FType.NUM: [],
            },
            item_ftypemeta={
                FType.CAT: self.net.embedding_map.item_cat_cols,
                # TODO: assume for now that all num feats are item-related (else, need extra book-keeping)
                FType.NUM: list(self.net.num_meta.items()) if hasattr(self.net, 'num_meta') else [],
            },
            context_ftypemeta={
                FType.CAT: self.net.embedding_map.context_cat_cols,
                # FType.NUM: [],
            },
            batch_size=batch_size)

    def get_loss(self) -> tf.Tensor:
        """
        Calculates the pair-loss between a positive and negative interaction
        :param input_xn_pair_d: dictionary of feature names to category codes
            for a pos/neg pair of interactions
        :return: scalar loss
        """

        with tf.name_scope('model'):
            # Split up input into pos & neg interaction
            pos_input_d = {k.split(TAG_DELIM, 1)[-1]: v for k, v in self.input_pair_d.items()
                           if k.startswith(USER_VAR_TAG + TAG_DELIM)
                           or k.startswith(POS_VAR_TAG + TAG_DELIM)
                           or k.startswith(CONTEXT_VAR_TAG + TAG_DELIM)
                           }
            neg_input_d = {k.split(TAG_DELIM, 1)[-1]: v for k, v in self.input_pair_d.items()
                           if k.startswith(USER_VAR_TAG + TAG_DELIM)
                           or k.startswith(NEG_VAR_TAG + TAG_DELIM)
                           or k.startswith(CONTEXT_VAR_TAG + TAG_DELIM)
                           }

            with tf.name_scope('positive'):
                pos_score = tf.identity(self.forward(
                    pos_input_d), name='pos_score')
            with tf.name_scope('negative'):
                neg_score = tf.identity(self.forward(
                    neg_input_d), name='neg_score')

            with tf.name_scope('loss'):
                # TODO: this should be tied to the sampling technique
                # Note: Hard coded BPR loss for now
                # loss_bpr = tf.subtract(1., tf.sigmoid(
                #     pos_score - neg_score), name='bpr')
                # return tf.reduce_mean(loss_bpr, name='bpr_mean')

                # loss_hinge = tf.clip_by_value(neg_score - pos_score + 1.,
                #                               clip_value_min=0.,
                #                               clip_value_max=999.,
                #                               name='hinge')
                # return tf.reduce_mean(loss_hinge, name='hinge_mean')

                loss_softplus = tf.log1p(tf.exp(neg_score - pos_score + 1.))
                return tf.reduce_mean(loss_softplus, name='softplus_mean')

    def training(self, loss) -> tf.Operation:
        """
        :param loss: scalar loss
        :return: training operation
        """
        loss_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss_tot = loss + loss_reg
        tf.summary.scalar('loss', loss)  # monitor loss w/o reg
        tf.summary.scalar('loss_reg', loss_reg)

        global_step = tf.get_variable(
            'global_step', shape=[], trainable=False,
            initializer=tf.constant_initializer(0))

        train_op = self.optimizer.minimize(loss_tot, global_step=global_step)
        return train_op
