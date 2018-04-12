import tensorflow as tf
from tophat.constants import *
from tophat.data import (InteractionsSource, InteractionsDerived,
                         FeatureSource,
                         TrainDataLoader,
                         )
from tophat.embedding import EmbeddingMap
from tophat.nets.bilinear import BilinearNet
from tophat.tasks.factorization import FactorizationTask
from tophat.losses import PairLossFn, NAMED_LOSSES
from tophat.sampling.pair_sampler import PairSampler
from typing import Dict, List, Optional, Union

# TODO: having trouble doing proper inheritance with the shady property
XN_SRC = Union[InteractionsSource, InteractionsDerived]


class FactorizationTaskWrapper(object):
    def __init__(self,
                 loss_fn: Union[str, PairLossFn],
                 sample_method: str,
                 interactions: XN_SRC,
                 group_features: Optional[
                     Dict[FGroup, List[FeatureSource]]] = None,
                 specific_feature: Optional[Dict[FGroup, bool]] = None,
                 nonnegs: Optional[XN_SRC] = None,
                 context_cols: Optional[List[str]] = None,
                 existing_cats_d: Optional[Dict[str, List[Any]]] = None,
                 embedding_map: Optional[EmbeddingMap] = None,
                 embedding_map_kwargs: Optional = None,
                 batch_size: Optional[int] = None,
                 sample_prefetch: Optional[int] = 10,
                 optimizer: Optional[tf.train.Optimizer] =
                    tf.train.AdamOptimizer(learning_rate=0.001),
                 seed: Optional[int] = 322,
                 name: Optional[str] = None,
                 ):

        self.name = name or f'{self.__class__.__name__}_{hex(id(self))}'
        self.seed = seed
        self.sample_method = sample_method
        self.sample_prefetch = sample_prefetch
        self.loss_fn = NAMED_LOSSES[loss_fn] if isinstance(loss_fn, str) \
            else loss_fn
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.data_loader = TrainDataLoader(
            interactions_train=interactions,
            group_features=group_features,
            specific_feature=specific_feature,
            context_cols=context_cols,
            batch_size=batch_size,
            existing_cats_d=existing_cats_d,
        )

        self.embedding_map = embedding_map or EmbeddingMap(
            cats_d=self.data_loader.cats_d,
            **embedding_map_kwargs,
        )

        self.net = BilinearNet(
            embedding_map=self.embedding_map,
            user_cat_cols=self.data_loader.user_cat_cols,
            item_cat_cols=self.data_loader.item_cat_cols,
            context_cat_cols=self.data_loader.context_cat_cols,
        )

        self.task = FactorizationTask(
            net=self.net,
            batch_size=self.batch_size,
            loss_fn=self.loss_fn,
            item_col=self.data_loader.item_col,  # only needed for k-OS
            optimizer=self.optimizer,
            name=self.data_loader.name or self.name,
        )

        non_neg_df = nonnegs.load().data if nonnegs else None
        self.sampler = PairSampler.from_data_loader(
                self.data_loader,
                self.task.input_pair_d,
                self.batch_size,
                method=self.sample_method,
                model=self.task,
                seed=self.seed,
                non_negs_df=non_neg_df,
            )
        # TODO: manually adding misc first violation (maybe find a cleaner way)
        if self.sample_method == 'adaptive_warp':  # or kos loss
            self.task.input_pair_d[f'{MISC_TAG}.first_violator_inds'] = \
                tf.placeholder(tf.int32, shape=[self.batch_size],
                               name=f'{MISC_TAG}.first_violator_inds_input')

        self.dataset = tf.data.Dataset.from_generator(
            self.sampler.__iter__,
            {k: v.dtype for k, v in self.task.input_pair_d.items()},
            {k: v.shape for k, v in self.task.input_pair_d.items()},) \
            .prefetch(self.sample_prefetch)

        self.input_pair_d_via_iter = self.dataset.make_one_shot_iterator() \
            .get_next()

        # Change out our legacy placeholders with this dataset iter
        self.task.input_pair_d = self.input_pair_d_via_iter

        # Get our training operations
        self.loss = self.task.get_loss()
        self.train_op = self.task.training(self.loss)

    def __len__(self):
        return len(self.data_loader.interactions_df)

    @property
    def steps_per_epoch(self):
        return len(self) / self.batch_size









