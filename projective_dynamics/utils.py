import os

def save_off_mesh(V, F, filename):
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def check_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


def fast_dense_plus_sparse_times_dense(a, A, b, d, weight):
    """
    Equivalent to: a += weight * (A * b[:, d]) for column d only.

    Args:
        a (np.ndarray): Dense matrix to update, shape (rows, D)
        A (scipy.sparse.spmatrix): Sparse matrix, shape (rows, cols)
        b (np.ndarray): Dense matrix, shape (cols, D)
        d (int): Column index of b to use
        weight (float): Scalar multiplier
    """
    # Use the CSC/CSR format for efficient iteration
    A = A.tocsr()
    col_d = b[:, d]  # column vector, shape (cols,)

    for i in range(A.shape[0]):  # iterate rows of A
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        for idx in range(row_start, row_end):
            j = A.indices[idx]
            a[i, d] += A.data[idx] * col_d[j] * weight
