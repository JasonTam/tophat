import numpy as np
import scipy.sparse as sp


def dropcols_coo(csr_mat: sp.csr_matrix, idx_to_drop):
    """
    Drop columns of sparse matrix
    http://stackoverflow.com/questions/23966923/delete-columns-of-matrix-of-csr-format-in-python
    """
    idx_to_drop = np.unique(idx_to_drop)
    coo_mat = csr_mat.tocoo()
    keep = ~np.in1d(coo_mat.col, idx_to_drop)

    coo_mat.data = coo_mat.data[keep]
    coo_mat.row = coo_mat.row[keep]
    coo_mat.col = coo_mat.col[keep]

    # decrement column indices
    coo_mat.col -= idx_to_drop.searchsorted(coo_mat.col)
    coo_mat._shape = (coo_mat.shape[0], coo_mat.shape[1] - len(idx_to_drop))
    return coo_mat.tocsr()


def get_row_nz(csr_mat: sp.csr_matrix, row_ind: int):
    """faster than csr_mat.get_row.nonzero()[-1]"""
    start_idx = csr_mat.indptr[row_ind]
    stop_idx = csr_mat.indptr[row_ind + 1]
    return csr_mat.indices[start_idx:stop_idx]


def get_row_nz_data(csr_mat: sp.csr_matrix, row_ind: int):
    """faster than csr_mat.get_row.nonzero()[-1]"""
    start_idx = csr_mat.indptr[row_ind]
    stop_idx = csr_mat.indptr[row_ind + 1]
    nz = csr_mat.indices[start_idx:stop_idx]
    data = csr_mat.data[start_idx:stop_idx]
    return nz, data
