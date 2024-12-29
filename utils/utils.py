# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import time
import functools
import numpy as np
import struct
from numpy.linalg import matrix_rank
from numpy import save, count_nonzero
from scipy.sparse import csr_matrix

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
    assert K <= points.shape[0]
    print("Storing ", points.shape[0], "interpolation points")
    if extension == '.bin':
        with open(fileName + 'F' + str(F) + colName + str(K)+ "_points" + str(points.shape[0]) + extension, 'wb') as doc0:
            doc0.write(struct.pack("<i", points.shape[0]))  # write a 4 byte integer in little endian
            doc0.write(struct.pack("<i", 1))  # write a 4 byte integer in little endian
            for k in range(points.shape[0]):
                value = points[k]
                doc0.write(struct.pack("<d", value))  # write a double precision (8 byte) in little endian
        doc0.close()

    if extension == '.npy':  # important in case we want to compare parts of the stored components as matrices
        save(fileName + str(F) + 'K' + str(K)+ "_points" + str(points.shape[0]), points)


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


def store_matrix(fileName, mat, d1, d2, extension='.bin'):

    assert mat.shape == (d1, d2)

    if extension == '.bin':
        with open(fileName + extension, 'wb') as doc0:
            doc0.write(struct.pack("<i", d2))  # write a 4 byte integer in little endian
            doc0.write(struct.pack("<i", d1))  # write a 4 byte integer in little endian
            for i in range(d1):
                for j in range(d2):
                    value = mat[i, j]
                    doc0.write(struct.pack("<d", value))  # write a double precision (8 byte) in little endian
        doc0.close()

    if extension == '.npy':  # important in case we want to compare parts of the stored components as matrices
        save(fileName, mat)


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


def write_tensor_to_bin_colmajor(tensor, filename):
    # Get the tensor's dimensions (N, Kp, 3)
    N, Kp, channels = tensor.shape

    # Ensure the tensor is 3D with shape (K, N, 3)
    if channels != 3:
        raise ValueError("The tensor must have 3 channels as the last dimension")

    # Convert the tensor to column-major order (Fortran style)
    tensor_col_major = np.asfortranarray(tensor)

    # Open the file in binary write mode
    with open(filename, 'wb') as f:
        # Write the dimensions as unsigned 32-bit integers
        f.write(np.array([N, Kp, channels], dtype=np.uint32).tobytes())

        # Write the tensor data in column-major order
        f.write(tensor_col_major.tobytes())

def read_sparse_matrix_from_bin(filename):
    with open(filename, "rb") as f:
        # Read dimensions
        rows = struct.unpack('<i', f.read(4))[0]
        cols = struct.unpack('<i', f.read(4))[0]

        # Read non-zero count
        nnz = struct.unpack('<i', f.read(4))[0]

        # Read row indices, column indices, and values
        row_indices = []
        col_indices = []
        values =[]
        for v in range(nnz):
            row_indices.append(struct.unpack('<i', f.read(4))[0] )
            col_indices.append(struct.unpack('<i', f.read(4))[0])
            values.append(struct.unpack('<d', f.read(8))[0])
        # row_indices = np.fromfile(f, dtype=np.int32, count=nnz)
        # col_indices = np.fromfile(f, dtype=np.int32, count=nnz)
        # values = np.fromfile(f, dtype=np.float64, count=nnz)  # Adjust dtype to match PDScalar in C++

    # Reconstruct sparse matrix in COO format
    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(rows, cols))
    return sparse_matrix


def read_mesh_file(file_path):
    """
    Reads a .mesh file and returns vertices, tetrahedra, and triangles as NumPy arrays.

    Parameters:
        file_path (str): Path to the .mesh file.

    Returns:
        tuple: Numpy arrays of vertices, tetrahedra, and triangles.
               Returns None if there is an error in reading the file.
    """
    try:
        vertices, tets, tris = [], [], []
        current_array = None

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                # Identify the section and prepare to read the corresponding data
                if line.startswith('Vertices'):
                    current_array = vertices
                    num_expected = int(next(file).strip())  # The next line should state the number of vertices
                    continue
                elif line.startswith('Tetrahedra'):
                    current_array = tets
                    num_expected = int(next(file).strip())
                    continue
                elif line.startswith('Triangles'):
                    current_array = tris
                    num_expected = int(next(file).strip())
                    continue

                # Skip empty lines or any line that doesn't fit into the above categories
                if not line or current_array is None:
                    continue

                # Parse and store data
                parts = line.split()
                if current_array is vertices:
                    # Expect x, y, z coordinates and one attribute (usually ignored)
                    if len(parts) >= 4:
                        current_array.append([float(parts[0]), float(parts[1]), float(parts[2])])
                else:
                    # For tets or tris, expect vertex indices and one attribute (usually ignored)
                    if len(parts) >= 4:
                        current_array.append([int(p) - 1 for p in parts[:-1]])  # Convert to zero-based index

        # Convert lists to numpy arrays
        vertices = np.array(vertices, dtype=float) if vertices else np.array([], dtype=float)
        tets = np.array(tets, dtype=int) if tets else np.array([], dtype=int)
        tris = np.array(tris, dtype=int) if tris else np.array([], dtype=int)

        return vertices, tets, tris

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None



