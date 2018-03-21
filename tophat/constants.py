"""
Some constant values
"""
from typing import *
from enum import Enum

# For reproducibility
SEED = 322

# Commonly used tags
USER_VAR_TAG = 'user'
POS_VAR_TAG = 'pos'
NEG_VAR_TAG = 'neg'
CONTEXT_VAR_TAG = 'context'
MISC_TAG = 'misc'
TAG_DELIM = '.'


class DummyEnum(Enum):
    """
    Enum-like aliasing class 
    where the enums are mostly interchangeable with their values
    """

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        return self.value.__eq__(other) or super().__eq__(other)

    def __hash__(self):
        return self.value.__hash__()


class FType(DummyEnum):
    """
    Feature Type
    """
    CAT = 'categorical'
    NUM = 'numerical'


class FGroup(DummyEnum):
    """
    Feature Group
    """
    USER = 'user'
    ITEM = 'item'
    CONTEXT = 'context'


# Dictionary of FType to list of feature names optionally with dimension
FtypeMeta = Dict[FType, Union[List[str], List[Tuple[str, int]]]]
