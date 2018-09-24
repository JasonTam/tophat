import itertools as it
from tophat.tasks.base import BaseTask
from tophat import losses
from tophat.nets.bilinear import *
from tophat.utils.ph_conversions import *


class FactorizationTask(BaseTask):
    """ Factorization Task

    Args:
        net: Prediction network
        batch_size: Batch size for fitting
        optimizer: Training optimizer object
        seed: Seed for random state
        item_col: name of item column -- used to get the number of items
            which is used for k-OS loss
        name: name of the model (to be used for the scope)
    """

    def __init__(self,
                 net: BilinearNet,
                 batch_size,
                 loss_fn: losses.PairLossFn = losses.softplus_loss,
                 optimizer: tf.train.Optimizer = tf.train.AdamOptimizer(
                     learning_rate=0.001),
                 input_pair_d: Optional[Dict[str, tf.Tensor]] = None,
                 seed=SEED,
                 item_col: Optional[str] = None,
                 name: Optional[str] = None,
                 ):

        super().__init__(batch_size=batch_size, seed=seed, name=name)

        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.input_pair_d: Dict[str, tf.Tensor] = None

        self.forward = self.net.forward

        # Make Placeholders according to our cats
        with tf.name_scope('placeholders'):
            self.input_pair_d = input_pair_d or self.get_pair_dict(
                self.batch_size)

        # for k-OS loss
        if item_col:
            self.n_items = len(self.net.embedding_map.cats_d[item_col])

    def get_fwd_dict(self, batch_size: int = None):
        """Gets the placeholders required for the forward prediction of a
        single interaction

        Args:
            batch_size: Optional batch_size if known before-hand

        Returns:
            Dictionary of placeholders

        """

        return fwd_dict_via_cats(
            it.chain(*self.net.cat_cols.values()),
            batch_size)

    def get_pair_dict(self, batch_size: int = None, extra_dim: bool = False):
        """Gets the placeholders required for the forward prediction of a
        pair of interactions

        Args:
            batch_size: Optional batch_size if known before-hand
            extra_dim: Optional extra dimension if each observation contains many samples to be aggregated

        Returns:
            Dictionary of placeholders

        """

        return pair_dict_via_ftypemeta(
            user_ftypemeta={
                FType.CAT: self.net.cat_cols[FGroup.USER],
                # FType.NUM: [],
            },
            item_ftypemeta={
                FType.CAT: self.net.cat_cols[FGroup.ITEM],
                # TODO: assume for now that all num feats are item-related
                #   (else, need extra book-keeping)
                FType.NUM: list(self.net.num_meta.items())
                if hasattr(self.net, 'num_meta') else [],
            },
            context_ftypemeta={
                FType.CAT: self.net.cat_cols[FGroup.CONTEXT],
                # FType.NUM: [],
            },
            batch_size=batch_size,
            extra_dim=extra_dim,
        )

    def get_loss(self) -> tf.Tensor:
        """Calculates the pair-loss between a positive and negative interaction

        Returns:
            Loss operation

        """

        def input_by_prefix(d, prefixes):
            """Filter input dictionary by prefix condition on key"""
            return {k.split(TAG_DELIM, 1)[-1]: v
                    for k, v in d.items()
                    if k.startswith(tuple(prefixes))}

        with tf.name_scope(f'task_{self.name}'):
            # Split up input into pos & neg interaction
            shared_prefixes = {
                USER_VAR_TAG + TAG_DELIM,
                CONTEXT_VAR_TAG + TAG_DELIM,
            }
            pos_prefixes = {POS_VAR_TAG + TAG_DELIM}.union(shared_prefixes)
            neg_prefixes = {NEG_VAR_TAG + TAG_DELIM}.union(shared_prefixes)

            pos_input_d = input_by_prefix(self.input_pair_d, pos_prefixes)
            neg_input_d = input_by_prefix(self.input_pair_d, neg_prefixes)

            with tf.name_scope('positive'):
                pos_score = tf.identity(self.forward(
                    pos_input_d), name='pos_score')
            with tf.name_scope('negative'):
                neg_score = tf.identity(self.forward(
                    neg_input_d), name='neg_score')

            first_violation = self.input_pair_d.get(
                f'{MISC_TAG}.first_violator_inds', None)

            with tf.name_scope('loss'):
                # Note: this could be tied to the sampling technique
                return self.loss_fn(pos_score, neg_score,
                                    first_violation, self.n_items,
                                    )

    def training(self, loss) -> tf.Operation:
        """Makes the training operation and attaches some summary values

        Args:
            loss: Loss operation

        Returns:
            Training operation

        """

        loss_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss_tot = loss + loss_reg

        tf.summary.scalar(f'{self.name}/loss', loss)  # loss w/o reg
        tf.summary.scalar(f'{self.name}/loss_reg', loss_reg)

        with tf.variable_scope('global'):
            task_step = tf.get_variable(
                f'{self.name}_step', shape=[], trainable=False,
                initializer=tf.constant_initializer(0))

        train_op = self.optimizer.minimize(loss_tot, global_step=task_step)
        return train_op
