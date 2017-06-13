import numpy as np
import scipy.sparse as sp


def dropcols_coo(M: sp.csr_matrix, idx_to_drop):
    """
    Drop columns of sparse matrix
    http://stackoverflow.com/questions/23966923/delete-columns-of-matrix-of-csr-format-in-python
    """
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()


def get_row_nz(csr_mat: sp.csr_matrix, row_ind: int):
    """faster than csr_mat.get_row.nonzero()[-1]"""
    start_idx = csr_mat.indptr[row_ind]
    stop_idx = csr_mat.indptr[row_ind + 1]
    return csr_mat.indices[start_idx:stop_idx]