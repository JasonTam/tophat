import numpy as np
import scipy.sparse as sp
from tophat.utils.sparse_utils import get_row_nz
from tophat.sampling.utils import neg_samp_bsearch
from typing import Sequence


def sample_uniform(n_items: int, batch_size: int = 1, n_neg: int = 1):
    """Sample negatives uniformly over entire catalog of items
    This is fast, but there is a chance of accidentally sampling a positive
    (See `sample_uniform_verified` to prevent this caveat)
    
    Args:
        n_items: number of items in catalog to sample from
        batch_size: number of samples to get
        n_neg: number of negatives to sample per positive

    Returns:
        Array with shape [batch_size] of random items as negatives

    """
    return np.random.randint(n_items, size=[batch_size, n_neg], dtype=np.uint32)


def sample_uniform_verified(
        n_items: int,
        xn_csr: sp.csr_matrix,
        user_inds_batch: Sequence[int],
        n_neg: int = 1,
):
    """Sample negatives uniformly over entire catalog of items
    Ensures that the neg samples are not known positives
    Note: This can be much slower than `sample_uniform` 

    Args:
        n_items: number of items in catalog to sample from
        xn_csr: sparse interaction matrix
        user_inds_batch: The users of the batch
            (used to lookup positives for verification)
            `batch_size` is assumed to be the number of users provided here
        n_neg: number of negatives to sample per positive

    Returns:
        Array with shape [batch_size] of random items as negatives

    """

    batch_size = len(user_inds_batch)
    neg_item_inds_batch = np.empty([batch_size, n_neg], dtype=np.uint32)
    for i, user_ind in enumerate(user_inds_batch):
        user_pos_item_inds = get_row_nz(xn_csr, user_ind)
        neg_inds = neg_samp_bsearch(user_pos_item_inds, n_items, n_neg)
        neg_item_inds_batch[i] = neg_inds

    return neg_item_inds_batch


def sample_uniform_ordinal(
        n_items: int,
        xn_csr: sp.csr_matrix,
        user_inds_batch: Sequence[int],
        pos_item_inds_batch: Sequence[int],
        n_neg: int = 1,
):
    """With ordinal tier verification

        Args:
            n_items: number of items in catalog to sample from
            xn_csr: sparse matrix of interaction tiers
                (a negative interaction will never be sampled to be paired with
                a positive interaction of a higher tier
                Ex. Give more important interactions higher values in this 
                matrix)
            user_inds_batch: The users of the batch
            pos_item_inds_batch: The positive items of the batch
            n_neg: number of negatives to sample per positive
    """

    batch_size = len(user_inds_batch)
    neg_item_inds_batch = np.empty([batch_size, n_neg], dtype=np.uint32)
    for i, (user_ind, pos_item_ind) in enumerate(zip(user_inds_batch, pos_item_inds_batch)):
        pos_item_val = float(xn_csr[user_ind, pos_item_ind])
        user_pos_item_inds = get_row_nz(xn_csr[user_ind] >= pos_item_val, 0)
        neg_inds = neg_samp_bsearch(user_pos_item_inds, n_items, n_neg)
        neg_item_inds_batch[i] = neg_inds

    return neg_item_inds_batch
