import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer, dropout, batch_norm
import numpy as np
from tiefrex.utils_xn import preset_interactions, kernel_via_xn_sets, muls_via_xn_sets
from typing import Iterable, Dict, Tuple
from collections import defaultdict


class EmbeddingMap(object):
    def __init__(self,
                 data_loader,
                 embedding_dim: int=16,
                 l2_bias: float=0.,
                 l2_emb: float=0.,
                 seed=322,
                 zero_init_rows: Dict[str, Iterable[int]]=None,
                 vis_specific_embs=True,
                 feature_weights_d: Dict[str, float]=None,
                 ):
        """
        :param data_loader: Expecting tiefrex.data.TrainDataLoader
        :param embedding_dim: number of dimensions in each embedding matrix (number of latent factors)
        :param l2_bias: l2 regularization scale (0 to disable) for bias (typically 0)
        :param l2_emb: l2 regularization scale (0 to disable) for embeddings
        :param zero_init_rows: if provided, we will zero out these particular records
            This is used to simulate cold start items
            Zero them out right away -- and they should remain zero due to regularization
                except when the particular entry is chosen as the sample negative :(
            (this is only done on embs since biases are initialized with 0 anyway)
        """
        self.seed = seed
        self.data_loader = data_loader
        self.cats_d = data_loader.cats_d
        self.user_cat_cols = data_loader.user_cat_cols
        self.item_cat_cols = data_loader.item_cat_cols
        self.context_cat_cols = data_loader.context_cat_cols

        self.l2_bias = l2_bias
        self.l2_emb = l2_emb
        self.reg_bias = tf.contrib.layers.l2_regularizer(scale=self.l2_bias)
        self.reg_emb = tf.contrib.layers.l2_regularizer(scale=self.l2_emb)

        self.embedding_dim = embedding_dim
        # Note: possibly need an emb for NaN code
        #     (can be index 0, and we will always add 1 to our codes)
        #     else, it should map to 0's tensor
        self.embeddings_d = {}
        self.biases_d = {}

        if feature_weights_d is None:
            self.feature_weights_d = defaultdict(lambda feat: 1.)
        else:
            self.feature_weights_d = feature_weights_d

        with tf.variable_scope('embeddings'):
            self.embeddings_d = {
                feat_name: tf.get_variable(
                    name=f'{feat_name}_embs',
                    shape=[len(cats), embedding_dim],
                    initializer=tf.random_normal_initializer(
                        mean=0., stddev=1. / self.embedding_dim,
                        seed=self.seed),
                    regularizer=self.reg_emb
                )
                for feat_name, cats in self.cats_d.items()
            }
        if zero_init_rows is not None:
            for k, v in zero_init_rows.items():
                z = np.ones([len(self.cats_d[k]), embedding_dim],
                            dtype=np.float32)
                z[v] = False
                self.embeddings_d[k] *= tf.constant(z)

        with tf.variable_scope('biases'):
            self.biases_d = {
                feat_name: tf.get_variable(
                    name=f'{feat_name}_biases',
                    shape=[len(cats)],
                    initializer=tf.zeros_initializer(),
                    regularizer=self.reg_bias
                )
                for feat_name, cats in self.cats_d.items()
            }

        # TODO: numerical specific factors for user (theta_u)
        self.vis_specific_embs = vis_specific_embs
        if self.vis_specific_embs:
            K2 = self.embedding_dim
            with tf.variable_scope('visual'):
                self.user_vis = tf.get_variable(  # vbpr: theta_u
                    name='user_vis',
                    # have K' = K (n_visual_factors = n_factors)
                    shape=[len(self.cats_d[data_loader.user_col]), K2],
                    initializer=tf.random_normal_initializer(
                        mean=0., stddev=1. / K2, seed=self.seed),
                    regularizer=self.reg_emb,
                )

    def look_up(self, input_xn_d
                ) -> Tuple[
        Dict[str, tf.Tensor],  # user
        Dict[str, tf.Tensor],  # item
        Dict[str, tf.Tensor],  # context
        Dict[str, tf.Tensor],  # biases (no hierarchy)
        # TODO: maybe better to have Dict[{user,item,context}-->Dict[featname, emb]] etc
    ]:
        """
        :param input_xn_d: dictionary of feature names to category codes
            for a single interaction
        """
        embeddings_user = lookup_wrapper(
            self.embeddings_d, input_xn_d, self.user_cat_cols,
            'user_lookup', name_tmp='{}_emb',
            feature_weights_d=self.feature_weights_d,
        )
        embeddings_item = lookup_wrapper(
            self.embeddings_d, input_xn_d, self.item_cat_cols,
            'item_lookup', name_tmp='{}_emb',
            feature_weights_d=self.feature_weights_d,
        )
        embeddings_context = lookup_wrapper(
            self.embeddings_d, input_xn_d, self.context_cat_cols,
            'context_lookup', name_tmp='{}_emb',
            feature_weights_d=self.feature_weights_d,
        )
        biases = lookup_wrapper(
            self.biases_d, input_xn_d, self.user_cat_cols + self.item_cat_cols + self.context_cat_cols,
            # # TODO: Temp: No user bias (kinda unconventional)
            # list(set(self.user_cat_cols).union(set(self.item_cat_cols)) - {'ops_user_id'}),
            'bias_lookup', name_tmp='{}_bias',
            feature_weights_d=self.feature_weights_d,
        )

        return embeddings_user, embeddings_item, embeddings_context, biases


def lookup_wrapper(emb_d: Dict[str, tf.Tensor],
                   input_xn_d: Dict[str, tf.Tensor],
                   cols: Iterable[str],
                   scope: str, name_tmp: str='{}',
                   feature_weights_d: Dict[str, float]=None,
                   ) -> Dict[str, tf.Tensor]:
    """
    Embedding lookup for each categorical feature
    Can be stacked downstream to yield a tensor
        ie) `tf.stack(list(looked_up.values()), axis=-1)`
    """
    if not cols:
        return {}
    with tf.name_scope(scope):
        looked_up = {feat_name: tf.nn.embedding_lookup(
            emb_d[feat_name], input_xn_d[feat_name],
            name=name_tmp.format(feat_name))
            for feat_name in cols}
        if feature_weights_d is not None:
            for feat_name, tensor in looked_up.items():
                if feat_name in feature_weights_d:
                    looked_up[feat_name] = tf.multiply(
                        tensor, feature_weights_d[feat_name],
                        name=f'{name_tmp.format(feat_name)}_weighted')

    return looked_up


class BilinearNet(object):
    def __init__(self,
                 embedding_map: EmbeddingMap,
                 interaction_type='inter',
                 ):
        self.embedding_map = embedding_map
        self.interaction_type = interaction_type

    def forward(self, input_xn_d: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Forward inference step to score a user-item interaction
        :param input_xn_d: dictionary of feature names to category codes
            for a single interaction
        """

        # Handle sparse (embedding lookup of categorical features)
        embeddings_user, embeddings_item, embeddings_context, biases = \
            self.embedding_map.look_up(input_xn_d)

        embs_all = {**embeddings_user,
                    **embeddings_item,
                    **embeddings_context}

        fields_d = {
            'user': self.embedding_map.user_cat_cols,
            'item': self.embedding_map.item_cat_cols,
            'context': self.embedding_map.context_cat_cols,
        }

        interaction_sets = preset_interactions(
            fields_d, interaction_type=self.interaction_type)

        with tf.name_scope('interaction_model'):
            contrib_dot = kernel_via_xn_sets(interaction_sets, embs_all)
            # bias for cat feature factors
            contrib_bias = tf.add_n(list(biases.values()), name='contrib_bias')

        score = tf.add_n([contrib_dot, contrib_bias], name='score')

        return score


class BilinearNetWithNum(object):
    def __init__(self,
                 embedding_map: EmbeddingMap,
                 num_meta: Dict[str, int]=None,
                 l2_vis: float=0.,
                 ruin=True,  # use ruining's formulation or our own
                 interaction_type='inter',
                 ):
        self.ruin = ruin
        self.embedding_map = embedding_map
        self.interaction_type = interaction_type
        self.num_meta = num_meta or {}
        # Params for numerical features
        # embedding matrix (for each numerical feature) -- fully connected layer
        self.l2_vis = l2_vis
        self.W_fc_num_d = {}
        self.b_fc_num_d = {}  # bias for fully connected
        self.b_num_factor_d = {}
        self.b_num_d = {}  # vbpr paper uses this shady bias matrix (beta')
        with tf.name_scope('numerical_reduction'):

            self.reg_vis = tf.contrib.layers.l2_regularizer(scale=self.l2_vis)
            K2 = self.embedding_map.embedding_dim
            for feat_name, dim_numerical in self.num_meta.items():
                self.W_fc_num_d[feat_name] = tf.get_variable(  # vbpr: E
                    name=f'{feat_name}_fc_embedder',
                    shape=[dim_numerical, K2],
                    initializer=tf.random_normal_initializer(stddev=1. / dim_numerical),
                    regularizer=self.reg_vis,
                )
                if not self.ruin:
                    self.b_fc_num_d[feat_name] = tf.get_variable(  # bias for E (not in paper)
                        name=f'{feat_name}_fc_bias',
                        shape=[self.embedding_map.embedding_dim],
                        initializer=tf.zeros_initializer(),
                    )
                    self.b_num_factor_d[feat_name] = tf.get_variable(  # just a scalar
                        name=f'{feat_name}_bias',
                        shape=[1],
                        initializer=tf.zeros_initializer(),
                    )
                else:
                    self.b_num_d[feat_name] = tf.get_variable(  # vbpr: beta'
                        name=f'{feat_name}_beta_prime',
                        shape=[dim_numerical],
                        initializer=tf.random_normal_initializer(stddev=1. / dim_numerical),
                        regularizer=self.reg_vis,
                    )

    def forward(self, input_xn_d: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Forward inference step to score a user-item interaction
        :param input_xn_d: dictionary of feature names to category codes
            for a single interaction
        """

        # Handle sparse (embedding lookup of categorical features)
        embeddings_user, embeddings_item, embeddings_context, biases = \
            self.embedding_map.look_up(
            input_xn_d)
        if self.embedding_map.vis_specific_embs:
            emb_user_vis = tf.nn.embedding_lookup(
                self.embedding_map.user_vis, input_xn_d[self.embedding_map.data_loader.user_col],
                name='user_vis_emb')
        else:
            emb_user_vis = None

        # Handle dense (fully connected reduction of dense features)
        # TODO: assume for now that all num feats are item-related (else, need extra book-keeping)
        user_num_cols = []
        if self.ruin:
            item_num_cols = []
        else:
            item_num_cols = list(self.num_meta.keys())
        if self.ruin:
            num_emb_d = {
                feat_name: tf.matmul(  # vbpr: theta_i
                    input_xn_d[feat_name], self.W_fc_num_d[feat_name], name='item_vis_emb')
                # + self.b_fc_num_d[feat_name]  # fc bias (not in vbpr paper)
                for feat_name in self.num_meta.keys()
            }
        else:
            num_emb_d = {
                feat_name: tf.matmul(  # vbpr: theta_i
                    input_xn_d[feat_name], self.W_fc_num_d[feat_name], name='item_vis_emb')
                + self.b_fc_num_d[feat_name]  # fc bias (not in vbpr paper)
                for feat_name in self.num_meta.keys()
            }

        embeddings_item.update(num_emb_d)  # TODO: temp assume num are item features (not vbpr)

        embs_all = {**embeddings_user,
                    **embeddings_item,
                    **embeddings_context}

        fields_d = {
            'user': self.embedding_map.user_cat_cols + user_num_cols,
            'item': self.embedding_map.item_cat_cols + item_num_cols,
        }

        interaction_sets = preset_interactions(
            fields_d, interaction_type=self.interaction_type)

        with tf.name_scope('interaction_model'):
            contrib_dot = kernel_via_xn_sets(interaction_sets, embs_all)
            # bias for cat feature factors
            if len(biases.values()):
                contrib_bias = tf.add_n(list(biases.values()), name='contrib_bias')
            else:
                contrib_bias = tf.zeros_like(contrib_dot, name='contrib_bias')

            if self.b_num_factor_d.values():
                # bias for num feature factors
                contrib_bias += tf.add_n(list(self.b_num_factor_d.values()))
            # NOTE: vbpr paper uses a bias matrix beta that we take a dot product with original numerical
            if self.b_num_d:
                contrib_vis_bias = tf.add_n(  # vbpr: beta * f
                    [tf.reduce_sum(
                        tf.multiply(input_xn_d[feat_name], self.b_num_d[feat_name]),
                        1, keep_dims=False
                    ) for feat_name in self.num_meta.keys()],
                    name='contrib_vis_bias'
                )
            else:
                contrib_vis_bias = tf.zeros_like(contrib_bias, name='contrib_vis_bias')

            # TODO: manually create visual interaction
            if len(num_emb_d):
                contrib_vis_dot = tf.add_n([
                    tf.reduce_sum(tf.multiply(emb_user_vis, num_emb), 1, keep_dims=False)  # theta_u.T * theta_i
                    for feat_name, num_emb in num_emb_d.items()
                ], name='contrib_vis_dot')
            else:
                contrib_vis_dot = tf.zeros_like(contrib_bias, name='contrib_vis_dot')

            score = tf.add_n([contrib_dot, contrib_bias, contrib_vis_dot, contrib_vis_bias],
                             name='score')
        return score


class BilinearNetWithNumFC(object):
    def __init__(self,
                 embedding_map: EmbeddingMap,
                 num_meta: Dict[str, int]=None,
                 l2=1e0,
                 interaction_type='inter',
                 ):
        """
        POC in replacing the inner product potion with FC layers
        References:
            He, Xiangnan, et al. "Neural collaborative filtering." 
            Proceedings of the 26th International Conference on World Wide Web. 
            International World Wide Web Conferences Steering Committee, 2017.
            
            Xiangnan He and Tat-Seng Chua (2017). 
            Neural Factorization Machines for Sparse Predictive Analytics. 
            In Proceedings of SIGIR '17, Shinjuku, Tokyo, Japan, August 07-11, 2017.
        """
        self.embedding_map = embedding_map
        self.interaction_type = interaction_type
        self.num_meta = num_meta or {}
        self.regularizer = l2_regularizer(l2)

    def forward(self, input_xn_d: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Forward inference step to score a user-item interaction
        :param input_xn_d: dictionary of feature names to category codes
            for a single interaction
        """

        # Handle sparse (embedding lookup of categorical features)
        embeddings_user, embeddings_item, embeddings_context, biases = self.embedding_map.look_up(
            input_xn_d)
        # bias not used

        num_emb_d = {
            feat_name: input_xn_d[feat_name] for feat_name in self.num_meta.keys()
        }

        embs_all = {**embeddings_user,
                    **embeddings_item,
                    **embeddings_context,
                    **num_emb_d}

        fields_d = {
            'user': self.embedding_map.user_cat_cols,
            'item': self.embedding_map.item_cat_cols,
        }

        interaction_sets = preset_interactions(
            fields_d, interaction_type=self.interaction_type)

        with tf.name_scope('interaction_model'):
            V = muls_via_xn_sets(interaction_sets, embs_all)
            f_bi = tf.add_n([node for s, node in V.items() if len(s) > 1],
                            name='f_bi')

        with tf.name_scope('deep'):
            # x = tf.concat([embs_all[k] for k in sorted(embs_all)], axis=1)
            x = tf.identity(f_bi, name='x')

            bn0 = batch_norm(x, decay=0.9)
            drop0 = dropout(bn0, 0.5)

            fc1 = fully_connected(drop0, 8, activation_fn=tf.nn.relu,
                                  weights_regularizer=self.regularizer)
            bn1 = batch_norm(fc1, decay=0.9)
            drop1 = dropout(bn1, 0.8)
            fc2 = fully_connected(drop1, 4, activation_fn=tf.nn.relu,
                                  weights_regularizer=self.regularizer)
            bn2 = batch_norm(fc2, decay=0.9)
            drop2 = dropout(bn2, 0.8)
            contrib_f = tf.squeeze(
                fully_connected(drop2, 1, activation_fn=None),
                name='score')

            contrib_bias = tf.add_n(list(biases.values()), name='contrib_bias')

        score = tf.add_n([contrib_f, contrib_bias], name='score')

        return score

