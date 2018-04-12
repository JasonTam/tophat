import tensorflow as tf
from abc import ABCMeta, abstractmethod
from typing import Optional


class BaseTask(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 batch_size: Optional[int] = None,
                 seed: Optional[int] = 0,
                 name: Optional[int] = None,
                 ):
        self.batch_size = batch_size
        self.seed = seed
        self.name = name or f'{self.__class__.__name__}_{hex(id(self))}'

    @abstractmethod
    def get_fwd_dict(self, batch_size: int = None):
        pass

    @abstractmethod
    def get_loss(self) -> tf.Tensor:
        pass

    @abstractmethod
    def training(self, *_) -> tf.Operation:
        pass
