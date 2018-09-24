import numpy as np
import scipy.sparse as sp
from typing import Sequence, Callable
from tophat.utils.sparse_utils import get_row_nz_data
from tophat.sampling.utils import neg_samp_bsearch


def sample_adaptive(
        n_items: int,
        max_sampled: int,
        score_fn: Callable,
        user_inds_batch: Sequence[int],
        pos_item_inds_batch: Sequence[int],
        use_first_violation: bool = False,
        xn_csr: sp.csr_matrix = None,
        return_n_samp: bool = False,
):
    """Uses the forward prediction of `self.model` to adaptively sample
    the first, or most violating negative candidate
    
    Note: for true WARP [1]_ sampling with k-OS loss, we need to also return
    the number of samples it took to reach the first violation
    to pass into our loss function. 
    (set `use_first_violation=True` and `return_n_samp=True` )

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
        use_first_violation: If True, the sampled negative will be the
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
        return_n_samp: If True, also return the number of samples to reach
            the first violation. Requires `use_first_violation` to be True.

    Returns:
        Array with shape [batch_size] of random items as negatives

    References:
        .. [1] Weston, Jason, Hector Yee, and Ron J. Weiss. "Learning to rank
           recommendations with the k-order statistic loss." Proceedings of the
           7th ACM conference on Recommender systems. ACM, 2013.

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
            user_neg_item_inds = neg_samp_bsearch(
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

        # For the users with no sampled violations, set violation index to inf
        #   this will cause the loss weight to be 0 (no update)
        neg_item_inds_batch = neg_item_inds[
            range(len(neg_item_inds)), first_violator_inds
        ]

        first_violator_inds[~violations[
            range(batch_size), first_violator_inds]
        ] = 2**31 - 1  # (our int32 "infinity")

        if return_n_samp:
            return neg_item_inds_batch, first_violator_inds
    else:
        # Get the worst offender
        neg_item_inds_batch = neg_item_inds[
            range(batch_size), np.argmax(neg_cand_scores, axis=1)
        ]

    return neg_item_inds_batch
