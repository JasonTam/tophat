import numpy as np


def neg_samp_bsearch(pos_inds: np.array, n_items: int, n_samp: int = 32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered

    Reference:
        https://medium.com/@2j/negative-sampling-in-numpy-18a9ad810385

    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds

