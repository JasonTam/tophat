import numpy as np
import scipy.sparse as sp
from typing import Sequence, Union, Dict, Callable
from tophat.utils.sparse_utils import get_row_nz, get_row_nz_data


def neg_samp(pos_inds: np.array, n_items: int, n_samp: int=32):
    """Samples negatives given an ordered array of positive indices to exclude
    from sampling
    
    Args:
        pos_inds: index of positive items to exclude from sampling
            Note: These indices must be ordered!
            As this method uses binary search
        n_items: total number of items in catalog
        n_samp: number of random samples to produce

    Returns:
        neg_inds: index of sampled negative items
        
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds


def sample_adaptive(
        n_items: int,
        max_sampled: int,
        score_fn: Callable,
        user_inds_batch: Sequence[int],
        pos_item_inds_batch: Sequence[int],
        use_first_violation: bool=False,
        xn_csr: sp.csr_matrix=None,
):
    """Uses the forward prediction of `self.model` to adaptively sample
    the first, or most violating negative candidate
    
    Note: for true WARP [4]_ sampling, we would need to also return 
    the number of samples it took to reach the first violation
    to pass into our loss function. Also `use_first_violation=True`.

    Args:
        n_items: number of items in catalog to sample from
        max_sampled: max number of negative candidates to sample per user
        score_fn: function that scores user-item pairs 
            (usually the forward inference pass of network)
        user_inds_batch: The users of the batch
            (used to score candidate negatives)
        pos_item_inds_batch: The positive items of the batch
            (used to find violations -- only in the case where 
            `use_first_violation` is True)
        use_first_violation: If True, the sampled negative will be be the
            first negative candidate to score over 1+score(positive). If
            there are no such violations, the last candidate is used.
            If False, use the worst offender
            (negative candidate with the highest score)
        xn_csr: sparse matrix of positive interactions
                If the data is boolean, this is simply nonpositive verification
                If this contains numerical data,
                ordinal tiered sampling will be done.
                (a negative interaction will never be sampled to be paired with
                a positive interaction of a higher tier
                Ex. Give more important interactions higher values in this 
                matrix)

    Returns:
        Array with shape [batch_size] of random items as negatives

    References:
        .. [4] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie:
           Scaling up to large vocabulary image annotation." IJCAI.
           Vol. 11. 2011.
           
    Todo:
        Number of samples from `use_first_violation` is not yet returned and 
        not handled in WARP loss yet (downstream)
        

    """

    batch_size = len(user_inds_batch)  # NOT max_sampled

    if xn_csr is None:
        neg_item_inds = np.random.randint(
            n_items, size=[batch_size, max_sampled])
    else:  # Ordinal verification
        neg_item_inds = np.empty([batch_size, max_sampled], dtype=int)
        pos_item_vals = xn_csr[user_inds_batch, pos_item_inds_batch]

        for i, (user_ind, cur_pos_item_val) in enumerate(
                zip(user_inds_batch, pos_item_vals.A1)):
            # Get all positive interactions for this user
            user_pos_item_inds, user_pos_item_vals = get_row_nz_data(
                xn_csr, user_ind)

            # Filter interactions of same or higher tier than current positive
            user_pos_item_inds = user_pos_item_inds[
                user_pos_item_vals >= cur_pos_item_val]
            user_neg_item_inds = neg_samp(
                user_pos_item_inds, n_items, max_sampled)

            neg_item_inds[i, :] = user_neg_item_inds

    # These have shape = (batch_size, max_sampled)
    neg_cand_scores = score_fn(
        user_inds=np.tile(user_inds_batch[:, None], max_sampled).flatten(),
        item_inds=neg_item_inds.flatten(),
    ).reshape([-1, max_sampled])

    if use_first_violation:
        pos_scores = score_fn(
            user_inds=user_inds_batch,
            item_inds=pos_item_inds_batch,
        )
        pos_scores_tile = np.tile(pos_scores[:, None], max_sampled)
        violations = (neg_cand_scores > pos_scores_tile - 1)  # hinge
        # Get index of the first violation
        first_violator_inds = np.argmax(violations, axis=1)

        # For the users with no violations, set first violation to last ind
        first_violator_inds[~violations[
            range(batch_size), first_violator_inds]
        ] = max_sampled - 1

        neg_item_inds_batch = neg_item_inds[
            range(len(neg_item_inds)), first_violator_inds
        ]
    else:
        # Get the worst offender
        neg_item_inds_batch = neg_item_inds[
            range(batch_size), np.argmax(neg_cand_scores, axis=1)
        ]

    return neg_item_inds_batch

