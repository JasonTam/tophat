import tensorflow as tf
import itertools
from tiefrex.constants import *
from typing import List, Dict, Iterable, Sized


class EmbeddingMap(object):
    def __init__(self,
                 cats_d: Dict[str, Sized],
                 user_feat_cols: List[str],
                 item_feat_cols: List[str],
                 embedding_dim: int=16,
                 l2: float=1e-5,
                 seed=SEED,
                 ):
        """
        :param cats_d: dictionary of feature name -> possible categorical values (catalog/vocabulary)
            (only the size of the catalog is needed)
        :param user_feat_cols: names of user features
        :param item_feat_cols: names of item features
        :param embedding_dim: number of dimensions in each embedding matrix (number of latent factors)
        :param l2: l2 regularization scale (0 to disable)
        """
        self.seed = seed
        self.cats_d = cats_d
        self.user_feat_cols = user_feat_cols
        self.item_feat_cols = item_feat_cols

        self.l2 = l2
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2)

        self.embedding_dim = embedding_dim
        # Note: possibly need an emb for NaN code
        #     (can be index 0, and we will always add 1 to our codes)
        #     else, it should map to 0's tensor

        with tf.variable_scope('embeddings'):
            self.embeddings_d = {
                feat_name: tf.get_variable(
                    name=f'{feat_name}_embs',
                    shape=[len(cats), embedding_dim],
                    initializer=tf.random_normal_initializer(
                        mean=0., stddev=1. / self.embedding_dim, seed=self.seed),
                    regularizer=self.regularizer
                )
                for feat_name, cats in self.cats_d.items()
            }

        with tf.variable_scope('biases'):
            self.biases_d = {
                feat_name: tf.get_variable(
                    name=f'{feat_name}_biases',
                    shape=[len(cats)],
                    initializer=tf.zeros_initializer()
                )
                for feat_name, cats in self.cats_d.items()
            }

    def look_up(self, input_xn_d):
        """
        :param input_xn_d: dictionary of feature names to category codes
            for a single interaction
        """
        with tf.name_scope('user_lookup'):
            self.embeddings_user = tf.stack([
                tf.nn.embedding_lookup(self.embeddings_d[feat_name], input_xn_d[feat_name],
                                       name=f'{feat_name}_lookedup')
                for feat_name in self.user_feat_cols
            ], axis=-1)

        with tf.name_scope('item_lookup'):
            self.embeddings_item = tf.stack([
                tf.nn.embedding_lookup(self.embeddings_d[feat_name], input_xn_d[feat_name],
                                       name=f'{feat_name}_lookedup')
                for feat_name in self.item_feat_cols
            ], axis=-1)

        with tf.name_scope('bias_lookup'):
            self.biases = tf.stack([
                tf.nn.embedding_lookup(
                    self.biases_d[feat_name], input_xn_d[feat_name])
                for feat_name in self.user_feat_cols + self.item_feat_cols
            ], axis=-1)

        return self.embeddings_user, self.embeddings_item, self.biases


class FactModel(object):

    def __init__(self,
                 embedding_map: EmbeddingMap,
                 # loss_type: str='bpr',
                 l2: float=1e-5,
                 intra_field: bool=False,
                 optimizer: tf.train.Optimizer=tf.train.AdamOptimizer(),
                 seed=SEED,
                 ):
        """
        :param l2: l2 regularization scale (0 to disable)
        :param intra_field: flag to include intra-field interactions
        :param optimizer: tf optimizer
        """

        self.seed = seed

        self.embedding_map = embedding_map

        self.l2 = l2
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2)

        self.intra_field = intra_field
        self.optimizer = optimizer

    def get_fwd_dict(self, batch_size):
        return fwd_dict_via_cats(
            self.embedding_map.cats_d.keys(),
            batch_size)

    def forward(self, input_xn_d) -> tf.Tensor:
        """
        Forward inference step to score a user-item interaction
        :param input_xn_d: dictionary of feature names to category codes
            for a single interaction
        """

        embeddings_user, embeddings_item, biases = self.embedding_map.look_up(
            input_xn_d)

        with tf.name_scope('interaction_model'):
            if self.intra_field:
                embeddings = tf.concat(
                    [embeddings_user, embeddings_item], axis=2)
                prod = tf.matmul(embeddings, embeddings, transpose_a=True)
            else:
                prod = tf.matmul(
                    embeddings_user, embeddings_item, transpose_a=True)

            contrib_dot = tf.reduce_sum(prod, axis=[1, 2], name='contrib_dot')
            contrib_bias = tf.reduce_sum(biases, axis=[1], name='contrib_bias')

            score = tf.add(contrib_dot, contrib_bias, name='score')

        return score

    def get_pair_dict(self, batch_size):
        return pair_dict_via_cols(
            self.embedding_map.user_feat_cols,
            self.embedding_map.item_feat_cols,
            batch_size)

    def get_loss(self, input_xn_pair_d) -> tf.Tensor:
        """
        Calculates the pair-loss between a postitive and negative interaction
        :param input_xn_pair_d: dictionary of feature names to category codes
            for a pos/neg pair of interactions
        :return: scalar loss
        """
        with tf.name_scope('model'):
            # Split up input into pos & neg interaction
            pos_input_d = {k.split(TAG_DELIM, 1)[-1]: v for k, v in input_xn_pair_d.items()
                           if k.startswith(USER_VAR_TAG + TAG_DELIM)
                           or k.startswith(POS_VAR_TAG + TAG_DELIM)}
            neg_input_d = {k.split(TAG_DELIM, 1)[-1]: v for k, v in input_xn_pair_d.items()
                           if k.startswith(USER_VAR_TAG + TAG_DELIM)
                           or k.startswith(NEG_VAR_TAG + TAG_DELIM)}

            pos_score = tf.identity(self.forward(
                pos_input_d), name='pos_score')
            neg_score = tf.identity(self.forward(
                neg_input_d), name='neg_score')

            with tf.name_scope('loss'):
                # Note: Hard coded BPR loss for now
                loss_bpr = tf.subtract(1., tf.sigmoid(
                    pos_score - neg_score), name='bpr')
                return tf.reduce_mean(loss_bpr, name='bpr_mean')

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


def fwd_dict_via_cats(cat_keys: Iterable[str], batch_size: int) -> Dict[str, tf.Tensor]:
    """ Creates placeholders for forward inference
    :param cat_keys: feature names
    :param batch_size: 
    :return: dictionary of placeholders
    """
    input_forward_d = {
        feat_name: tf.placeholder(tf.int32, shape=[batch_size],
                                  name=f'{feat_name}_input')
        for feat_name in cat_keys
    }
    return input_forward_d


def pair_dict_via_cols(user_feat_cols: Iterable[str],
                       item_feat_cols: Iterable[str],
                       batch_size: int) -> Dict[str, tf.Tensor]:
    """ Creates placeholders for paired loss
    :param user_feat_cols: 
    :param item_feat_cols: 
    :param batch_size: 
    :return: 
    """
    input_pair_d = {
        **{f'{USER_VAR_TAG}.{feat_name}': tf.placeholder(
            tf.int32, shape=[batch_size], name=f'{USER_VAR_TAG}.{feat_name}_input')
            for feat_name in user_feat_cols},
        **{f'{POS_VAR_TAG}.{feat_name}': tf.placeholder(
            tf.int32, shape=[batch_size], name=f'{POS_VAR_TAG}.{feat_name}_input')
            for feat_name in item_feat_cols},
        **{f'{NEG_VAR_TAG}.{feat_name}': tf.placeholder(
            tf.int32, shape=[batch_size], name=f'{NEG_VAR_TAG}.{feat_name}_input')
            for feat_name in item_feat_cols}
    }
    return input_pair_d
