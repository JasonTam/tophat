import numpy as np
import scipy.sparse as sp
from tophat.utils.sparse_utils import get_row_nz
from tophat.sampling.utils import neg_samp_bsearch
from typing import Sequence


def sample_weighted(weights_cs: np.array,
                    batch_size: int = 1, n_neg: int = 1):
    """Sample negatives uniformly over entire catalog of items
    This is fast, but there is a chance of accidentally sampling a positive
    (See `sample_uniform_verified` to prevent this caveat)
    
    Args:
        weights_cs: array of item cumulative sampling weights
        batch_size: number of samples to get
        n_neg: number of negatives to sample per positive

    Returns:
        Array with shape [batch_size] of random items as negatives

    """

    r = np.random.rand(batch_size, n_neg)
    return np.searchsorted(weights_cs, r, side='right')
