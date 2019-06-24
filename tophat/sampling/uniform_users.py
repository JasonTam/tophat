import numpy as np
import scipy.sparse as sp
from tophat.utils.sparse_utils import get_row_nz
from typing import List, Collection


def sample_user_pos(
        user_inds_batch: Collection[int],
        pos_xn_csr: sp.csr_matrix,
        rand: np.array,
        *_):

    batch_size = len(user_inds_batch)
    rand_rats = rand.rand(batch_size)
    pos_l = []

    for user_ind, rat in zip(user_inds_batch, rand_rats):
        # TODO: `get_row_nz` repeated in sampling
        user_pos_item_inds = get_row_nz(pos_xn_csr, user_ind)
        user_pos_item = user_pos_item_inds[
            int(rat * len(user_pos_item_inds))]
        pos_l.append(user_pos_item)
    pos_item_inds_batch = np.array(pos_l)

    return pos_item_inds_batch


def sample_user_pos_weighted(
        user_inds_batch: Collection[int],
        pos_xn_csr: sp.csr_matrix,
        rand: np.array,
        cs_l: List[np.array],
):

    batch_size = len(user_inds_batch)
    rand_rats = rand.rand(batch_size)
    pos_l = []

    for user_ind, rat in zip(user_inds_batch, rand_rats):
        cs = cs_l[user_ind]
        pos_item_inds = get_row_nz(pos_xn_csr, user_ind)
        ind = np.searchsorted(cs, rat, side='right')
        user_pos_item = pos_item_inds[ind]
        pos_l.append(user_pos_item)
    pos_item_inds_batch = np.array(pos_l)

    return pos_item_inds_batch
