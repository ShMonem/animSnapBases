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




def delete_matching_column(matrix_lil, target_col_vector):
    """
    Deletes the first column from matrix_lil that matches target_col_vector.
    If the only matching column is the last column, return None.
    """
    matrix = matrix_lil.tocsc()
    target_col = target_col_vector.tocsr()

    cols_to_keep = []
    matched_indices = []

    for col in range(matrix.shape[1]):
        col_vector = matrix[:, col].tocsr()
        if (col_vector != target_col).nnz == 0:
            matched_indices.append(col)
        else:
            cols_to_keep.append(col)

    if not matched_indices:
        # No match found, return original
        return matrix_lil

    if matched_indices == [matrix.shape[1] - 1] and len(cols_to_keep) == matrix.shape[1] - 1:
        # The only match is the last column
        return None

    matrix_new = matrix[:, cols_to_keep].tolil()
    return matrix_new
