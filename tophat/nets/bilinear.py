import tensorflow as tf
from typing import Dict, Callable, List
from collections import ChainMap

from tophat.constants import FGroup
from tophat.embedding import EmbeddingMap
from tophat.utils.xn_utils import \
    preset_interactions, kernel_via_xn_sets, muls_via_xn_sets
from tophat.nets.fc import simple_fc


class BilinearNet(object):
    """Network for scoring interactions

    Args:
        embedding_map: Variables and metadata concerning categorical embeddings
        user_cat_cols: Name of user categorical feature columns
        item_cat_cols: Name of item categorical feature columns
        context_cat_cols: Name of context categorical feature columns
        interaction_type: Type of preset interaction
            One of {'intra', 'inter'}
    """

    def __init__(self,
                 embedding_map: EmbeddingMap,
                 user_cat_cols: List[str],
                 item_cat_cols: List[str],
                 context_cat_cols: List[str],
                 interaction_type='inter',
                 ):
        self.embedding_map = embedding_map
        self.cat_cols = {
            FGroup.USER: user_cat_cols,
            FGroup.ITEM: item_cat_cols,
            FGroup.CONTEXT: context_cat_cols,
        }
        self.interaction_type = interaction_type
        self.num_meta = {}

    def forward(self, input_xn_d: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Forward inference step to score a user-item interaction
        
        Args:
            input_xn_d: Dictionary of feature names to category codes
                for a single interaction

        Returns:
            Forward inference scoring operation

        """

        # Handle sparse (embedding lookup of categorical features)
        embs_by_group, biases = self.embedding_map.look_up(
            input_xn_d, self.cat_cols)
        embs_all = ChainMap(*embs_by_group.values())

        fields_d = {
            fg: self.cat_cols[fg]
            for fg in [FGroup.USER, FGroup.ITEM, FGroup.CONTEXT]
        }

        interaction_sets = preset_interactions(
            fields_d, interaction_type=self.interaction_type)

        with tf.name_scope('interactions'):
            contrib_dot = kernel_via_xn_sets(interaction_sets, embs_all)
            # bias for cat feature factors
            contrib_bias = tf.add_n(list(biases.values()), name='contrib_bias')

        score = tf.add_n([contrib_dot, contrib_bias], name='score')

        return score


class BilinearNetWithNum(BilinearNet):
    """Forward inference step to score a user-item interaction
    With the ability to handle numerical (visual) features based on [1]_

    Args:
        embedding_map:Variables and metadata concerning categorical embeddings
        num_meta: Metadata concerning numerical data
            `feature_name -> dimensionality of input`
        l2_vis: l2 regularization scale for visual embedding matrix
        ruin: If True, use the formulation of [1]_
            Else, use a modified formulation
        interaction_type: Type of preset interaction
            One of {'intra', 'inter'}

    References:
        .. [1] He, Ruining, and Julian McAuley. "VBPR: Visual Bayesian 
           Personalized Ranking from Implicit Feedback." AAAI. 2016.

    """
    def __init__(self,
                 embedding_map: EmbeddingMap,
                 user_cat_cols: List[str],
                 item_cat_cols: List[str],
                 context_cat_cols: List[str],
                 interaction_type: str = 'inter',
                 num_meta: Dict[str, int] = None,
                 l2_vis: float = 0.,
                 ruin: bool = True,
                 ):
        BilinearNet.__init__(self, embedding_map,
                             user_cat_cols,
                             item_cat_cols,
                             context_cat_cols,
                             interaction_type)

        self.ruin = ruin
        self.num_meta = num_meta or {}
        # Params for numerical features
        # embedding matrix for each numerical feature (fully connected layer)
        self.l2_vis = l2_vis
        self.W_fc_num_d = {}
        self.b_fc_num_d = {}  # bias for fully connected
        self.b_num_factor_d = {}
        self.b_num_d = {}  # vbpr paper uses this shady bias matrix (beta')
        with tf.name_scope('numerical_reduction'):

            self.reg_vis = tf.contrib.layers.l2_regularizer(scale=self.l2_vis)
            K2 = self.embedding_map.embedding_dim
            for feat_name, dim_numerical in self.num_meta.items():
                # vbpr: E
                self.W_fc_num_d[feat_name] = tf.get_variable(
                    name=f'{feat_name}_fc_embedder',
                    shape=[dim_numerical, K2],
                    initializer=tf.random_normal_initializer(
                        stddev=1. / dim_numerical),
                    regularizer=self.reg_vis,
                )
                if not self.ruin:
                    # bias for E (not in paper)
                    self.b_fc_num_d[feat_name] = tf.get_variable(
                        name=f'{feat_name}_fc_bias',
                        shape=[self.embedding_map.embedding_dim],
                        initializer=tf.zeros_initializer(),
                    )
                    # just a scalar
                    self.b_num_factor_d[feat_name] = tf.get_variable(
                        name=f'{feat_name}_bias',
                        shape=[1],
                        initializer=tf.zeros_initializer(),
                    )
                else:
                    # vbpr: beta'
                    self.b_num_d[feat_name] = tf.get_variable(
                        name=f'{feat_name}_beta_prime',
                        shape=[dim_numerical],
                        initializer=tf.random_normal_initializer(
                            stddev=1. / dim_numerical),
                        regularizer=self.reg_vis,
                    )

    def forward(self, input_xn_d: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Forward inference step to score a user-item interaction
        
        Args:
            input_xn_d: Dictionary of feature names to category codes
                for a single interaction

        Returns:
            Forward inference scoring operation

        """

        # Handle sparse (embedding lookup of categorical features)
        embs_by_group, biases = self.embedding_map.look_up(
            input_xn_d, self.cat_cols)

        if self.embedding_map.vis_emb_user_col:
            emb_user_vis = tf.nn.embedding_lookup(
                self.embedding_map.user_vis,
                input_xn_d[self.embedding_map.vis_emb_user_col],
                name='user_vis_emb')
        else:
            emb_user_vis = None

        # Handle dense (fully connected reduction of dense features)
        # TODO: assume for now that all num feats are item-related
        #   (else, need extra book-keeping)
        user_num_cols = []
        if self.ruin:
            item_num_cols = []
        else:
            item_num_cols = list(self.num_meta.keys())
        if self.ruin:
            num_emb_d = {
                feat_name: tf.matmul(  # vbpr: theta_i
                    input_xn_d[feat_name], self.W_fc_num_d[feat_name],
                    name='item_vis_emb')
                # + self.b_fc_num_d[feat_name]  # fc bias (not in vbpr paper)
                for feat_name in self.num_meta.keys()
            }
        else:
            num_emb_d = {
                feat_name: tf.matmul(  # vbpr: theta_i
                    input_xn_d[feat_name], self.W_fc_num_d[feat_name],
                    name='item_vis_emb')
                + self.b_fc_num_d[feat_name]  # fc bias (not in vbpr paper)
                for feat_name in self.num_meta.keys()
            }

        # TODO: temp assume num are item features (not vbpr)
        embs_by_group[FGroup.ITEM].update(num_emb_d)

        embs_all = ChainMap(*embs_by_group.values())

        fields_d = {
            FGroup.USER:
                self.cat_cols[FGroup.USER] + user_num_cols,
            FGroup.ITEM:
                self.cat_cols[FGroup.ITEM] + item_num_cols,
        }

        interaction_sets = preset_interactions(
            fields_d, interaction_type=self.interaction_type)

        with tf.name_scope('interactions'):
            contrib_dot = kernel_via_xn_sets(interaction_sets, embs_all)
            # bias for cat feature factors
            if len(biases.values()):
                contrib_bias = tf.add_n(list(biases.values()),
                                        name='contrib_bias')
            else:
                contrib_bias = tf.zeros_like(contrib_dot,
                                             name='contrib_bias')

            if self.b_num_factor_d.values():
                # bias for num feature factors
                contrib_bias += tf.add_n(list(self.b_num_factor_d.values()))
            # NOTE: vbpr paper uses a bias matrix beta that we take a
            #   dot product with original numerical
            if self.b_num_d:
                contrib_vis_bias = tf.add_n(  # vbpr: beta * f
                    [tf.reduce_sum(
                        tf.multiply(input_xn_d[feat_name],
                                    self.b_num_d[feat_name]),
                        1, keep_dims=False
                    ) for feat_name in self.num_meta.keys()],
                    name='contrib_vis_bias'
                )
            else:
                contrib_vis_bias = tf.zeros_like(contrib_bias,
                                                 name='contrib_vis_bias')

            # TODO: manually create visual interaction
            if len(num_emb_d):
                contrib_vis_dot = tf.add_n([
                    tf.reduce_sum(
                        # theta_u.T * theta_i
                        tf.multiply(emb_user_vis, num_emb), 1, keep_dims=False)
                    for feat_name, num_emb in num_emb_d.items()
                ], name='contrib_vis_dot')
            else:
                contrib_vis_dot = tf.zeros_like(contrib_bias,
                                                name='contrib_vis_dot')

            score = tf.add_n([contrib_dot,
                              contrib_bias,
                              contrib_vis_dot,
                              contrib_vis_bias],
                             name='score')
        return score


class BilinearNetWithNumFC(BilinearNet):
    """POC to replace the inner product potion with FC layers as described in
    [2]_ and [3]_

    Args:
        embedding_map: Variables and metadata concerning categorical embeddings
        num_meta: Metadata concerning numerical data
            `feature_name -> dimensionality of input`
        interaction_type: Type of preset interaction
            One of {'intra', 'inter'}
        deep_net_fn: function to create deep portion of network
        deep_reg: regularizer for deep portion of network

    References:
        .. [2] He, Xiangnan, et al. "Neural collaborative filtering." 
           Proceedings of the 26th International Conference on World Wide 
           Web. International World Wide Web Conferences Steering 
           Committee, 2017.
        
        .. [3] Xiangnan He and Tat-Seng Chua (2017). Neural Factorization 
           Machines for Sparse Predictive Analytics. In Proceedings of 
           SIGIR '17, Shinjuku, Tokyo, Japan, August 07-11, 2017.
    """

    def __init__(self,
                 embedding_map: EmbeddingMap,
                 user_cat_cols: List[str],
                 item_cat_cols: List[str],
                 context_cat_cols: List[str],
                 interaction_type: str = 'inter',
                 num_meta: Dict[str, int] = None,
                 deep_net_fn: Callable = simple_fc,
                 deep_reg=None,
                 ):
        BilinearNet.__init__(self, embedding_map,
                             user_cat_cols,
                             item_cat_cols,
                             context_cat_cols,
                             interaction_type)

        self.num_meta = num_meta or {}
        self.deep_net_fn = deep_net_fn
        self.deep_reg = deep_reg

    def forward(self, input_xn_d: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Forward inference step to score a user-item interaction
        
        Args:
            input_xn_d: Dictionary of feature names to category codes
                for a single interaction

        Returns:
            Forward inference scoring operation

        """

        # Handle sparse (embedding lookup of categorical features)
        embs_by_group, biases = self.embedding_map.look_up(
            input_xn_d, self.cat_cols)

        num_emb_d = {
            feat_name: input_xn_d[feat_name]
            for feat_name in self.num_meta.keys()
        }

        embs_all = ChainMap(*embs_by_group.values(), num_emb_d)

        fields_d = {
            fg: self.cat_cols[fg]
            for fg in [FGroup.USER, FGroup.ITEM, FGroup.CONTEXT]
        }

        interaction_sets = preset_interactions(
            fields_d, interaction_type=self.interaction_type)

        with tf.name_scope('interactions'):
            xn_muls = muls_via_xn_sets(interaction_sets, embs_all)
            # Bi-Interaction (actually, we allow for >=2 interactions)
            f_bi = tf.add_n([node for s, node in xn_muls.items()
                             if len(s) > 1],
                            name='f_bi')

        contrib_deep = tf.identity(
            self.deep_net_fn(f_bi, self.deep_reg, scope_name='deep'),
            name='contrib_deep')
        contrib_bias = tf.add_n(list(biases.values()), name='contrib_bias')

        score = tf.add_n([contrib_deep, contrib_bias], name='score')

        return score

