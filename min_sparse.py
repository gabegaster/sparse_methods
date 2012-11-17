import numpy as np
from scipy import sparse

def min_rows_sparse(X):
    if type(X) is not sparse.csc_matrix:
        X = X.tocsc()
    return np.array(map(min_sparse,map(X.getrow,xrange(X.shape[1]))))

def min_cols_sparse(X):
    if type(X) is not sparse.csc_matrix:
        X = X.tocsc()
    return np.array(map(min_sparse,map(X.getcol,xrange(X.shape[0]))))

def min_sparse(X):
    if type(X) is not sparse.csc_matrix:
        X = X.tocsc()
    m = X.data
    if len(m) == 0:
        return 0
    else:
        m = m.min()
    return m if X.getnnz() == X.size else min(m, 0)
