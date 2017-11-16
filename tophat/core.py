from tophat.ph_conversions import *
from tophat.nets import *
from typing import Dict


class FactModel(object):
    """ Factorization Model

    Args:
        net: Prediction network
        optimizer: Training optimizer object
        seed: Seed for random state
    """

    def __init__(self,
                 net,
                 optimizer: tf.train.Optimizer = tf.train.AdamOptimizer(
                     learning_rate=0.001),
                 seed=SEED,
                 ):

        self.seed = seed
        self.net = net
        self.optimizer = optimizer
        self.input_pair_d: Dict[str, tf.Tensor] = None

        self.forward = self.net.forward

        # Make Placeholders according to our cats
        with tf.name_scope('placeholders'):
            self.input_pair_d = self.get_pair_dict(
                self.net.embedding_map.data_loader.batch_size)

    def get_fwd_dict(self, batch_size: int=None):
        """Gets the placeholders required for the forward prediction of a
        single interaction

        Args:
            batch_size: Optional batch_size if known before-hand

        Returns:
            Dictionary of placeholders

        """

        return fwd_dict_via_cats(
            self.net.embedding_map.cats_d.keys(),
            batch_size)

    def get_pair_dict(self, batch_size: int=None):
        """Gets the placeholders required for the forward prediction of a
        pair of interactions

        Args:
            batch_size: Optional batch_size if known before-hand

        Returns:
            Dictionary of placeholders

        """

        return pair_dict_via_ftypemeta(
            user_ftypemeta={
                FType.CAT: self.net.embedding_map.user_cat_cols,
                # FType.NUM: [],
            },
            item_ftypemeta={
                FType.CAT: self.net.embedding_map.item_cat_cols,
                # TODO: assume for now that all num feats are item-related
                #   (else, need extra book-keeping)
                FType.NUM: list(self.net.num_meta.items())
                if hasattr(self.net, 'num_meta') else [],
            },
            context_ftypemeta={
                FType.CAT: self.net.embedding_map.context_cat_cols,
                # FType.NUM: [],
            },
            batch_size=batch_size)

    def get_loss(self) -> tf.Tensor:
        """Calculates the pair-loss between a positive and negative interaction

        Returns:
            Loss operation

        """

        with tf.name_scope('model'):
            # Split up input into pos & neg interaction
            pos_input_d = {k.split(TAG_DELIM, 1)[-1]: v
                           for k, v in self.input_pair_d.items()
                           if k.startswith(USER_VAR_TAG + TAG_DELIM)
                           or k.startswith(POS_VAR_TAG + TAG_DELIM)
                           or k.startswith(CONTEXT_VAR_TAG + TAG_DELIM)
                           }
            neg_input_d = {k.split(TAG_DELIM, 1)[-1]: v
                           for k, v in self.input_pair_d.items()
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

        Args:
            loss: Loss operation

        Returns:
            Training operation

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
