import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.tensorboard.plugins import projector
from tophat.data import TrainDataLoader
from tophat.metadata_proc import write_metadata_emb
from typing import Iterable, Dict, Tuple, Optional, List, Any
from collections import defaultdict


class EmbeddingMap(object):
    def __init__(self,
                 cats_d: Dict[str, List[Any]],
                 user_cat_cols: Iterable[str],
                 item_cat_cols: Iterable[str],
                 context_cat_cols: Iterable[str],
                 embedding_dim: int=16,
                 l1_bias: float=0.,
                 l2_bias: float=0.,
                 l1_emb: float=0.,
                 l2_emb: float=0.,
                 seed=322,
                 zero_init_rows: Optional[Dict[str, Iterable[int]]]=None,
                 feature_weights_d: Optional[Dict[str, float]]=None,
                 vis_emb_user_col: Optional[str]=None,
                 init_emb_d: Optional[Dict[str, tf.Tensor]]=None,
                 ):
        """Convenience container for embedding layers
        
        Args:
            cats_d: Dictionary of categories
            user_cat_cols: Name of user categorical feature columns
            item_cat_cols: Name of item categorical feature columns
            context_cat_cols: Name of context categorical feature columns
            embedding_dim: Size of embedding dimension
                (number of latent factors)
            l1_bias: l1 regularization scale for bias
                (0 to disable -- typically 0)
            l1_emb: l1 regularization scale for embeddings
                (0 to disable)
            l2_bias: l2 regularization scale for bias
                (0 to disable -- typically 0)
            l2_emb: l2 regularization scale for embeddings
                (0 to disable)
            seed: Seed for random state
            zero_init_rows: If provided, we will zero out these particular records
                This is used to simulate cold start items
                Zero them out right away
                    and they should remain zero due to regularization
                    except when the particular entry is chosen as the 
                    sample negative :(
                    (this is only done on embs since biases are initialized 
                    with 0 anyway)
            feature_weights_d: Feature weights by name
            vis_emb_user_col: If provided, give users an additional
                visual-specific embedding (the value of the argument will 
                denote the name of the user column)
            init_emb_d: Optional dictionary of weights to initialize embeddings
                Can be a subset of features, but needs to conform to the
                proper shape of the embedding. If there are categories missing
                from a feature's embedding initialization, please fill them in
                prior with some initialization scheme.
        """

        self.seed = seed
        self.cats_d = cats_d
        self.user_cat_cols = user_cat_cols
        self.item_cat_cols = item_cat_cols
        self.context_cat_cols = context_cat_cols

        # Regularization
        self.l1_bias = l1_bias
        self.l2_bias = l2_bias
        self.reg_bias = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=self.l1_bias, scale_l2=self.l2_bias)
        self.l1_emb = l1_emb
        self.l2_emb = l2_emb
        self.reg_emb = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=self.l1_emb, scale_l2=self.l2_emb)

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
            self.embeddings_d = {}

            for feat_name, cats in self.cats_d.items():
                if init_emb_d is not None and feat_name in init_emb_d:
                    # Initialize from passed-in weights
                    emb_init = init_emb_d[feat_name]
                    assert emb_init.shape == [len(cats), embedding_dim]
                    shape = None
                else:
                    # Nothing to load, just rand initialization
                    emb_init = tf.random_normal_initializer(
                        mean=0., stddev=1. / self.embedding_dim,
                        seed=self.seed)
                    shape = [len(cats), embedding_dim]
                self.embeddings_d[feat_name] = tf.get_variable(
                    name=f'{feat_name}_embs',
                    shape=shape,
                    initializer=emb_init,
                    regularizer=self.reg_emb
                )

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
        self.vis_emb_user_col = vis_emb_user_col
        if self.vis_emb_user_col:
            K2 = self.embedding_dim
            with tf.variable_scope('visual'):
                self.user_vis = tf.get_variable(  # vbpr: theta_u
                    name='user_vis',
                    # have K' = K (n_visual_factors = n_factors)
                    shape=[len(self.cats_d[self.vis_emb_user_col]), K2],
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
        """Looks up all the embeddings associated with a batch of interactions
        (user, item, context, and all biases)
        
        Args:
            input_xn_d: Dictionary of feature names to category codes
                for a single interaction
                
        Returns:
            Tuple of embeddings and biases
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
    """Embedding lookup for each categorical feature
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


def inits_via_avro(path_or_buf, cats: List[Any]
                   ) -> tf.Tensor:
    return inits_via_df(read_avro(path_or_buf), cats)


def read_avro(path_or_buf) -> pd.DataFrame:
    # TODO: this should come from some io package
    import fastavro as avro

    if isinstance(path_or_buf, str):
        buf = open(path_or_buf, 'rb')
    else:
        buf = path_or_buf

    reader = avro.reader(buf)
    df = pd.DataFrame(list(reader))
    try:
        buf.close()
    except AttributeError:
        print('Stream has no attribute close')

    return df


def inits_via_df(df: pd.DataFrame, cats: List[Any]) -> tf.Tensor:
    """Creates a tensor with initialization constants from a dataframe
    of preloaded weights. The tensor will have the correct shape as dictated
    by an input list of categories. Entries missing from the dataframe will
    be filled from a random normal distribution.

    Args:
        df: dataframe of input weights
        cats: categories (corresponds with the order of embeddings)

    Returns:

    """
    # Note: the loaded index is always str type
    factors = df.set_index('id').reindex(np.array(cats).astype(str)).factors

    # Fill in missing with rand norm init
    factors_arr = np.vstack(df.factors)
    samp_mean = np.mean(factors_arr)
    samp_std = np.std(factors_arr)
    emb_dim = factors_arr.shape[1]
    init_values = np.vstack(factors.apply(
        lambda x: x
        if hasattr(x, '__iter__')  # proxy for isnan
        else np.random.normal(loc=samp_mean, scale=samp_std, size=emb_dim)))

    emb_init = tf.constant(init_values, dtype='float32')

    return emb_init


class EmbeddingProjector(object):
    """Class to handle tensorboard embedding projections

    Args:
        embedding_map: Embedding map object
        summary_writer: Summary writer object
        config: Config params
    """

    def __init__(self,
                 embedding_map: EmbeddingMap,
                 summary_writer: tf.summary.FileWriter,
                 config):

        self.summary_writer = summary_writer
        feat_to_metapath = write_metadata_emb(
            embedding_map.cats_d,
            config.get('names'),
            config.get('log_dir'))

        self.projection_config = projector.ProjectorConfig()
        emb_proj_d = {}
        for feat_name, emb in embedding_map.embeddings_d.items():
            if feat_name in feat_to_metapath:
                emb_proj_d[feat_name] = self.projection_config.embeddings.add()
                emb_proj_d[feat_name].tensor_name = emb.name
                emb_proj_d[feat_name].metadata_path = feat_to_metapath[feat_name]

    def viz(self):
        # After the last step, lets save some embedding to viz later
        projector.visualize_embeddings(self.summary_writer,
                                       self.projection_config)