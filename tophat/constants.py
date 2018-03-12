"""
Some constant values
"""
from typing import Dict, Union, List, Tuple
from enum import Enum, auto

# For reproducibility
SEED = 322

# Commonly used tags
USER_VAR_TAG = 'user'
POS_VAR_TAG = 'pos'
NEG_VAR_TAG = 'neg'
CONTEXT_VAR_TAG = 'context'
MISC_TAG = 'misc'
TAG_DELIM = '.'


class FType(Enum):
    CAT = auto()
    NUM = auto()


# Dictionary of FType to list of feature names optionally with dimension
FtypeMeta = Dict[FType, Union[List[str], List[Tuple[str, int]]]]
