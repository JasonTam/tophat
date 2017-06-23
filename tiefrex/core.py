import tensorflow as tf
import itertools as it
from tiefrex.constants import *
from typing import List, Dict, Iterable, Sized, Set
from tiefrex.metadata_proc import write_metadata_emb
from tensorflow.contrib.tensorboard.plugins import projector


def preset_interactions(fields_d: Dict[str, Iterable[str]],
                        interaction_type: str='inter',
                        max_order: int=2
                        ) -> Iterable[frozenset]:
    """
    Convenience feature interaction planner for common interaction types
    :param fields_d: dictionary of group_name to iterable of feat_names that belong to that group
        ex) `fields_d = {'user': {'gender', 'age'}, 'item': {'brand', 'pcat', 'price'}}`
    :param interaction_type: preset type
    :param max_order: max order of interactions  
    :return: Iterable of interaction sets
    """
    if interaction_type == 'intra':  # includes intra field
        feature_pairs = it.chain(
            *(it.combinations(
                it.chain(*fields_d.values()), order)
                for order in range(2, max_order + 1)))
    elif interaction_type == 'inter':  # only inter field
        feature_pairs = it.product(*fields_d.values())
    else:
        raise ValueError

    return map(frozenset, feature_pairs)


def kernel_via_xn_sets(interaction_sets: Iterable[frozenset],
                       emb_d: Dict[str, tf.Tensor]) -> tf.Tensor:
    """
    Computes arbitrary order interaction terms
    Reuses lower order terms

    Differs from typical HOFM as we will reuse lower order embeddings
        (not actually sure if this is OK in terms of expressiveness)
        In theory, we're supposed to use a new param matrix for each order
            Much like how we use a bias param for order=1
    :param interaction_sets: interactions to create nodes for
    :param emb_d: dictionary of embedding tensors
        NOTE: would include an additional `order` key to match HOFM literature
    :return: 

    References:
        Blondel, Mathieu, et al. "Higher-Order Factorization Machines." 
            Advances in Neural Information Processing Systems. 2016.
    """
    # TODO: for now we assume that all dependencies of previous order are met

    interaction_sets = list(interaction_sets)
    # Populate xn nodes with single terms (order 1)
    xn_nodes: Dict[frozenset, tf.Tensor] = {
        frozenset({k}): v for k, v in emb_d.items()}
    unq_orders = set(map(len, interaction_sets))
    for order in range(min(unq_orders), max(unq_orders) + 1):
        with tf.name_scope(f'xn_order_{order}'):
            for xn in interaction_sets:
                if len(xn) == order:
                    xn_l = list(xn)
                    xn_nodes[xn] = tf.multiply(
                        xn_nodes[frozenset(xn_l[:-1])],  # cached portion (if ho)
                        xn_nodes[frozenset({xn_l[-1]})],  # last as new term
                        name='X'.join(xn)
                    )

    # Reduce nodes of order > 1
    contrib_dot = tf.add_n([
        tf.reduce_sum(node, 1, keep_dims=False)
        for s, node in xn_nodes.items() if len(s) > 1
    ], name='contrib_dot')
    return contrib_dot


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


class EmbeddingMap(object):
    def __init__(self,
                 data_loader,
                 embedding_dim: int=16,
                 l2: float=1e-5,
                 seed=SEED
                 ):
        """
        :param data_loader: Expecting tiefrex.data.TrainDataLoader
        :param embedding_dim: number of dimensions in each embedding matrix (number of latent factors)
        :param l2: l2 regularization scale (0 to disable)
        """
        self.seed = seed
        self.data_loader = data_loader
        self.cats_d = data_loader.cats_d
        self.user_feat_cols = data_loader.user_feat_cols
        self.item_feat_cols = data_loader.item_feat_cols

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

    def look_up(self, input_xn_d, stacked=False):
        """
        :param input_xn_d: dictionary of feature names to category codes
            for a single interaction
        :param stacked: if `True`, stack the embeddings of each group into a single tensor
        """
        with tf.name_scope('user_lookup'):
            self.embeddings_user = {
                feat_name:
                    tf.nn.embedding_lookup(self.embeddings_d[feat_name], input_xn_d[feat_name],
                                           name=f'{feat_name}_emb')
                for feat_name in self.user_feat_cols}
            if stacked:
                self.embeddings_user = tf.stack(list(self.embeddings_user.values()), axis=-1)

        with tf.name_scope('item_lookup'):
            self.embeddings_item = {
                feat_name:
                    tf.nn.embedding_lookup(self.embeddings_d[feat_name], input_xn_d[feat_name],
                                           name=f'{feat_name}_emb')
                for feat_name in self.item_feat_cols}

            if stacked:
                self.embeddings_item = tf.stack(list(self.embeddings_item.values()), axis=-1)

        with tf.name_scope('bias_lookup'):
            self.biases = {
                feat_name:
                    tf.nn.embedding_lookup(self.biases_d[feat_name], input_xn_d[feat_name],
                                           name=f'{feat_name}_bias')
                for feat_name in self.user_feat_cols + self.item_feat_cols}

            if stacked:
                self.biases = tf.stack(list(self.biases.values()), axis=-1)

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

        embs_all = {**embeddings_user, **embeddings_item}

        fields_d = {
            'user': self.embedding_map.user_feat_cols,
            'item': self.embedding_map.item_feat_cols,
        }

        interaction_sets = preset_interactions(
            fields_d, interaction_type='inter')

        with tf.name_scope('interaction_model'):
            contrib_dot = kernel_via_xn_sets(interaction_sets, embs_all)
            contrib_bias = tf.add_n(list(biases.values()), name='contrib_bias')
            score = tf.add(contrib_dot, contrib_bias, name='score')

        return score

    def get_pair_dict(self, batch_size):
        return pair_dict_via_cols(
            self.embedding_map.user_feat_cols,
            self.embedding_map.item_feat_cols,
            batch_size)

    def get_loss(self) -> tf.Tensor:
        """
        Calculates the pair-loss between a positive and negative interaction
        :param input_xn_pair_d: dictionary of feature names to category codes
            for a pos/neg pair of interactions
        :return: scalar loss
        """
        # Make Placeholders according to our cats
        with tf.name_scope('placeholders'):
            input_xn_pair_d = pair_dict_via_cols(
                self.embedding_map.data_loader.user_feat_cols, self.embedding_map.data_loader.item_feat_cols, batch_size=self.embedding_map.data_loader.batch_size)
            self.input_pair = input_xn_pair_d

        with tf.name_scope('model'):
            # Split up input into pos & neg interaction
            pos_input_d = {k.split(TAG_DELIM, 1)[-1]: v for k, v in input_xn_pair_d.items()
                           if k.startswith(USER_VAR_TAG + TAG_DELIM)
                           or k.startswith(POS_VAR_TAG + TAG_DELIM)}
            neg_input_d = {k.split(TAG_DELIM, 1)[-1]: v for k, v in input_xn_pair_d.items()
                           if k.startswith(USER_VAR_TAG + TAG_DELIM)
                           or k.startswith(NEG_VAR_TAG + TAG_DELIM)}

            with tf.name_scope('positive'):
                pos_score = tf.identity(self.forward(
                    pos_input_d), name='pos_score')
            with tf.name_scope('negative'):
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
