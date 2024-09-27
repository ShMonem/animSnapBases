# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import time
import functools
import numpy as np
import struct
from numpy.linalg import matrix_rank
from numpy import save, count_nonzero


def store_components(fileName, F, K, N, dim, basesTensor, extension='.bin',  colName='K'):
    """
    :param fileName: file to store data
    :param F: num of frames/snapshots used to compute the bases
    :param K: num of bases, in the nonlinearFun bases case 'Kp' is expected instead of 'K'
    :param N: either 'nVerts' for positions bases, or 'ep' for nonlinear case
    :param dim: =3 (x, y, z)
    :param basesTensor: of size (K, N, 3) or (Kp, ep, 3)
    :param extension: file type
    :return: stored file in the given path
    """
    assert basesTensor.shape == (K, N, dim)
    if extension == '.bin':
        with open(fileName + 'F' + str(F) + colName + str(K) + extension, 'wb') as doc0:
            doc0.write(struct.pack("<i", N))  # write a 4 byte integer in little endian
            doc0.write(struct.pack("<i", dim * K))  # write a 4 byte integer in little endian
            for d in range(dim):
                for k in range(K):
                    for i in range(N):
                        value = basesTensor[k, i, d]
                        doc0.write(struct.pack("<d", value))  # write a double precision (8 byte) in little endian
        doc0.close()

    if extension == '.npy':  # important in case we want to compare parts of the stored components as matrices
        save(fileName + str(F) + 'K' + str(K), basesTensor)


def testSparsity(mat):
    """
    :param name: bases type
    :param mat: expects tensor of size (:, :, 3)
    :param test_dim: always the third dim (x, y, z)
    :return: informs if the mat slices along test dimension are sparse
    """
    assert mat.shape[2] == 3
    # print('    ' + name + '...', end='', flush=True)
    sparPerList = []
    for l in range(3):
        sparPer = 1 - (count_nonzero(mat[:, :, l]) / mat[:, :, l].size)
        sparPerList.append(sparPer)
    if min(sparPerList) > 0.5:
        print("sparse, min %" + str(100*min(sparPerList)) + " zero entries.")
    else:
        print("... not sparse.")


def test_linear_dependency(mat, test_dim_range, expected_rank):
    """
        :param mat: expects tensor of size (:, :, 3)
        :param test_dim_range: typically for 3 (x,y,z)
        :param expected_rank: 'K' in case positions bases, and 'Kp' in case nonlinearFun bases
        :return: informs if the mat slices along test dimension are not linear independent
        """
    assert mat.shape[2] == 3

    for j in range(test_dim_range):
        if matrix_rank(mat[:, :, j]) == expected_rank:
            print(".. linear independent.")
        else:
            print("... not linear independent, with rank: " + str(matrix_rank(mat[:, :, j]))
                  + " != " + str(expected_rank) + ".")


def store_interpol_points_vector(fileName, F, K, points, extension='.bin', colName='K'):
    """
    :param fileName: file to store data
    :param F: num of frames/snapshots used to compute the bases
    :param K: num of bases blocks in the nonlinearFun bases
    :param points: array contains the indices at which the nonlinearFunc bases are computed
    :param extension: files type
    :return: stored file in the given path
    """
    assert points.shape[0] == K
    if extension == '.bin':
        with open(fileName + 'F' + str(F) + colName + str(K) + extension, 'wb') as doc0:
            doc0.write(struct.pack("<i", K))  # write a 4 byte integer in little endian
            doc0.write(struct.pack("<i", 1))  # write a 4 byte integer in little endian
            for k in range(K):
                value = points[k]
                doc0.write(struct.pack("<d", value))  # write a double precision (8 byte) in little endian
        doc0.close()

    if extension == '.npy':  # important in case we want to compare parts of the stored components as matrices
        save(fileName + str(F) + 'K' + str(K), points)


def store_vector(fileName, points, numPoints, extension='.bin'):

    assert points.shape[0] == numPoints

    if extension == '.bin':
        with open(fileName + '_' + str(numPoints) + extension, 'wb') as doc0:
            doc0.write(struct.pack("<i", numPoints))  # write a 4 byte integer in little endian
            doc0.write(struct.pack("<i", 1))  # write a 4 byte integer in little endian
            for k in range(numPoints):
                value = points[k]
                doc0.write(struct.pack("<d", value))  # write a double precision (8 byte) in little endian
        doc0.close()

    if extension == '.npy':  # important in case we want to compare parts of the stored components as matrices
        save(fileName + '_' + str(numPoints), points)


def check_matrix_properties(A):
    # Step 1: Check if the matrix is square
    rows, cols = A.shape
    if rows != cols:
        raise ValueError("Matrix is not square. A square matrix is required.")

    print("Square.")

    # Step 2: Check determinant (to check if the matrix is singular)
    det = np.linalg.det(A)
    if np.isclose(det, 0.0):
        raise ValueError("Matrix is singular (determinant is 0).")

    print(f"Determinant: {det}")

    # Step 3: Check the condition number (to check numerical stability)
    cond_number = np.linalg.cond(A)
    if cond_number > 1e12:  # Condition number is too high
        raise ValueError(f"Matrix has a high condition number ({cond_number}), which may cause numerical instability.")

    print(f"Condition number: {cond_number}")

    # Step 4: Check the rank (to check if the matrix is full rank)
    rank = np.linalg.matrix_rank(A)
    if rank != rows:
        raise ValueError("Matrix is rank-deficient. It may not be invertible or suitable for solving linear systems.")

    print(f"Rank: {rank} (Full rank: {rank == rows})")

    # Step 5: Check if the matrix is symmetric (optional)
    if np.allclose(A, A.T):
        print("Symmetric.")
    else:
        print("Matrix is not symmetric.")

    # Step 6: Check if the matrix is positive definite (optional)
    # try:
    #     np.linalg.cholesky(A)
    #     print("Matrix is positive definite.")
    # except np.linalg.LinAlgError:
    #     print("Matrix is not positive definite.")

    # New Step: Check for wide range of values in the matrix (ill-conditioning due to extreme values)
    singular_values = np.linalg.svd(A, compute_uv=False)  # Compute singular values
    max_singular_value = singular_values.max()
    min_singular_value = singular_values.min()

    # Check if singular values vary too much (large range)
    if max_singular_value / min_singular_value > 1e12:
        raise ValueError("Matrix has a wide range of singular values, indicating possible numerical instability.")

    print(f"Max singular value: {max_singular_value}")
    print(f"Min singular value: {min_singular_value}")
    print("Matrix passed all checks. It can be used to solve a linear system.")

# Decorator to measure and log the time of a function
# Global flag to check if we have already written to the file in this run
has_written = False
def log_time(filePath):
    def decorator(func):
        @functools.wraps(func)  # To keep the function metadata intact
        def wrapper(*args, **kwargs):
            global has_written

            # Check if the file exists and whether we have written in the current run
            if not has_written:
                mode = "w"  # First time, write mode (replace file)
                has_written = True  # Set flag to True after first write
            else:
                mode = "a"  # Subsequent writes, append mode

            start_time = time.time()  # Start timing
            result = func(*args, **kwargs)  # Run the actual function
            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time

            # Print to the screen
            print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")

            # Log the function name and time to a text file
            with open(filePath+"function_timings.txt", mode) as f:
                f.write(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.\n")

            return result
        return wrapper
    return decorator
