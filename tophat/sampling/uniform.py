import numpy as np
import scipy.sparse as sp
from tophat.utils.sparse_utils import get_row_nz
from typing import Sequence


def sample_uniform(n_items: int, batch_size: int = 1):
    """Sample negatives uniformly over entire catalog of items
    This is fast, but there is a chance of accidentally sampling a positive
    (See `sample_uniform_verified` to prevent this caveat)
    
    Args:
        n_items: number of items in catalog to sample from
        batch_size: number of samples to get

    Returns:
        Array with shape [batch_size] of random items as negatives

    """
    return np.random.randint(n_items, size=batch_size)


def sample_uniform_verified(
        n_items: int,
        xn_csr: sp.csr_matrix,
        user_inds_batch: Sequence[int],
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

    Returns:
        Array with shape [batch_size] of random items as negatives

    """

    neg_item_inds_batch = []
    for user_ind in user_inds_batch:
        user_pos_item_inds = get_row_nz(xn_csr, user_ind)
        neg_item_ind = np.random.randint(n_items)
        while neg_item_ind in user_pos_item_inds:
            neg_item_ind = np.random.randint(n_items)
        neg_item_inds_batch.append(neg_item_ind)
    return neg_item_inds_batch


def sample_uniform_ordinal(
        n_items: int,
        xn_csr: sp.csr_matrix,
        user_inds_batch: Sequence[int],
        pos_item_inds_batch: Sequence[int],
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
    """

    neg_item_inds_batch = []
    for user_ind, pos_item_ind in zip(user_inds_batch, pos_item_inds_batch):
        user_pos_item_inds = get_row_nz(xn_csr, user_ind)
        neg_item_ind = np.random.randint(n_items)
        while True:
            if neg_item_ind in user_pos_item_inds:
                # if we sampled a positive, make sure it is of lower tier
                neg_item_val = xn_csr[user_ind, neg_item_ind]
                pos_item_val = xn_csr[user_ind, pos_item_ind]
                if neg_item_val < pos_item_val:
                    break
            else:
                break
            neg_item_ind = np.random.randint(n_items)
        neg_item_inds_batch.append(neg_item_ind)
    return neg_item_inds_batch
