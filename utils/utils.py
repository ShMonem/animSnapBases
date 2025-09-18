# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import time
import functools
import numpy as np
import struct
from numpy.linalg import matrix_rank
from numpy import save, count_nonzero
from scipy.sparse import csr_matrix, load_npz
import shutil
import os
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


def load_vector_values(full_filename):
    """
    Reads a binary file storing an interpolated points vector.

    :param full_filename: Full path to the .bin file
    :return: NumPy array of stored points
    """
    with open(full_filename, 'rb') as f:
        num_points = struct.unpack('<i', f.read(4))[0]  # Read number of points
        _ = struct.unpack('<i', f.read(4))[0]  # Read and discard second int (usually 1)
        points = []
        for _ in range(num_points):
            val = struct.unpack('<d', f.read(8))[0]  # Read a double
            points.append(val)
    return np.array(points, dtype=np.float64)

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


def class_log_time(filePath):
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

def read_sparse_matrix(file_name, type, key=None):
    if type == ".bin":
        return read_sparse_matrix_from_bin(file_name)
    elif type == ".npz":
        if key is None:
            raise ValueError("Empty file or wrong key:", type)
        else:
            return np.load(file_name, allow_pickle=True)[key]
    else:
        raise ValueError("unknown sparse matrix file_type:" , type)

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


def read_obj(filename):
    vertices = []
    faces = []
    tetrahedrons = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append(list(map(float, parts[1:])))
            elif parts[0] == 'f':
                # Assumes that the OBJ file uses 1-based index
                faces.append([int(p.split('/')[0]) - 1 for p in parts[1:]])
    return np.array(vertices), np.array(faces)


def read_obj(filename):
    vertices = []
    faces = []
    tetrahedrons = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append(list(map(float, parts[1:])))

            elif parts[0] == 'f':
                # Parse face (triangles or quads); assumes 1-based index
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)

            elif parts[0] == 't':
                # Custom format for tetrahedra (1-based indices)
                tet = [int(idx) - 1 for idx in parts[1:5]]
                tetrahedrons.append(tet)

    return  np.array(vertices), np.array(faces, dtype=int) if faces else None, np.array(tetrahedrons, dtype=int) if tetrahedrons else None


def copy_and_delete_file(original_path, new_path):
    """
    Copies content from original_path to new_path and deletes the original file.

    Parameters:
        original_path (str): Path to the original file (e.g., 'file1.txt')
        new_path (str): Path to the new file (e.g., 'file2.txt')
    """
    # Copy file content
    shutil.copyfile(original_path, new_path)

    # Delete the original file permanently
    os.remove(original_path)

    print(f"Copied '{original_path}' to '{new_path}' and deleted the original.")


def tetrahedralize(V, F):
    tetgen_options = "pq1.2Y"  # or "pq1.414a0.01"
    from igl import boundary_facets, barycenter, winding_number, copyleft
    from igl.copyleft import tetgen

    TV, TT, TF = copyleft.tetgen.tetrahedralize(V, F, switches=tetgen_options)

    # if not success:
    #     print("[ERROR] Tetrahedralization failed.")
    #     return

    TT = TT[:, ::-1]  # reverse rows
    TF = TF[:, ::-1]

    BC = barycenter(TV, TT)
    W = winding_number(V, F, BC)

    inside = (W > 0.5)
    IT = TT[inside]

    G = boundary_facets(IT)
    G = G[:, ::-1]

    positions = TV
    tets = IT
    faces = G
    return positions, tets, faces


# Partitioning and heat method #
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import trimesh
import igl

def _lumped_mass_from_areas(n, idx, areas):
    # Accumulate per-element areas (or volumes) to vertices in idx (shape (m,k))
    Mdiag = np.zeros(n, dtype=float)
    per_vert = (areas[:, None] / idx.shape[1])
    np.add.at(Mdiag, idx.reshape(-1), np.repeat(per_vert, idx.shape[1], axis=1).ravel())
    return sp.diags(Mdiag, 0)

def _face_areas(V, F):
    # Triangle areas
    e1 = V[F[:,1]] - V[F[:,0]]
    e2 = V[F[:,2]] - V[F[:,0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

def _tet_volumes(V, T):
    # Tetra volumes
    v1 = V[T[:,1]] - V[T[:,0]]
    v2 = V[T[:,2]] - V[T[:,0]]
    v3 = V[T[:,3]] - V[T[:,0]]
    return np.abs(np.einsum('ij,ij->i', v1, np.cross(v2, v3))) / 6.0

def _safe_factor(A):
    # Robust factorization helper (A should be SPD)
    try:
        return spla.factorized(A.tocsc())
    except Exception:
        # Fall back to CG solve wrapper
        def cg_solve(b):
            x, _ = spla.cg(A, b, atol=1e-10, tol=1e-10, maxiter=2000)
            return x
        return cg_solve
# ------------------------------------------------------------
# Heat-method geodesics (surface)
# ------------------------------------------------------------
def heat_distance_surface(V, F, sources, t=None):
    """
    Heat-method geodesic distance on a triangle mesh from 'sources' (list/array of vertex ids).
    Returns: d (n,) geodesic distances (shifted so min at sources is ~0).
    """
    n = V.shape[0]
    if igl is None:
        # Fallback: graph-Dijkstra (edge-length weighted) – not a true heat method,
        # but keeps the API working if libigl is missing.
        import heapq
        G = sp.coo_matrix((np.linalg.norm(V[F[:,0]]-V[F[:,1]], axis=1), (F[:,0], F[:,1])), shape=(n,n))
        G = G + G.T
        # add the remaining edges (0-2) and (1-2)
        l02 = np.linalg.norm(V[F[:,0]]-V[F[:,2]], axis=1)
        l12 = np.linalg.norm(V[F[:,1]]-V[F[:,2]], axis=1)
        G = G + sp.coo_matrix((l02, (F[:,0], F[:,2])), shape=(n,n)) + sp.coo_matrix((l02, (F[:,2], F[:,0])), shape=(n,n))
        G = G + sp.coo_matrix((l12, (F[:,1], F[:,2])), shape=(n,n)) + sp.coo_matrix((l12, (F[:,2], F[:,1])), shape=(n,n))
        G = G.tocsr()
        INF = 1e30
        d = np.full(n, INF)
        hq = []
        for s in np.atleast_1d(sources):
            d[s] = 0.0
            heapq.heappush(hq, (0.0, s))
        while hq:
            dist_u, u = heapq.heappop(hq)
            if dist_u != d[u]: continue
            for v in G[u].indices:
                w = G[u, v]
                nd = dist_u + w
                if nd < d[v]:
                    d[v] = nd
                    heapq.heappush(hq, (nd, v))
        d -= d.min()
        return d

    # Use libigl operators
    L = igl.cotmatrix(V, F)          # negative semi-definite
    A = _face_areas(V, F)
    M = _lumped_mass_from_areas(V.shape[0], F, A)  # lumped mass

    # choose time step: t ~ mean edge length squared
    if t is None:
        # mean edge length
        E0 = V[F[:,1]] - V[F[:,0]]
        E1 = V[F[:,2]] - V[F[:,0]]
        el = np.r_[np.linalg.norm(E0, axis=1), np.linalg.norm(E1, axis=1),
                   np.linalg.norm(V[F[:,2]]-V[F[:,1]], axis=1)]
        t = (el.mean() ** 2)

    # Step 1: (M - t L) u = M δ   (note: L is negative, so (M - tL) is SPD)
    Aheat = (M - t * L)
    solve_heat = _safe_factor(Aheat)

    delta = np.zeros(n)
    src = np.atleast_1d(sources).astype(int)
    delta[src] = 1.0 / len(src)
    rhs_u = M @ delta
    u = solve_heat(rhs_u)

    # Step 2: compute normalized gradient field per face
    Gop = igl.grad(V, F)             # (3m x n)
    gu = (Gop @ u).reshape(-1, 3)    # per-face gradient
    # normalize per-face (avoid divide-by-zero)
    norms = np.linalg.norm(gu, axis=1, keepdims=True) + 1e-15
    X = -gu / norms                  # vector field

    # Step 3: divergence and Poisson solve  (-L) φ = div  (since L is cotmatrix (negative))
    # Divergence = -G^T (A_i * X_i), assemble per-face weighting
    # Build block-diag area weights for 3 components
    # X is (m,3); weight each face vector by its area, then flatten to (3m,)
    Xw = (X * A[:, None]).reshape(-1)  # (3m,)
    div = - (Gop.T @ Xw)  # (n,)

    K = (-L)  # SPD
    solve_poisson = _safe_factor(K)
    phi = solve_poisson(div)

    # Shift so sources are at ~0
    src = np.atleast_1d(sources).astype(int)
    phi -= phi[src].min()

    # If everything is <= 0 (0 at sources, negative elsewhere), flip sign
    if np.max(phi) <= 1e-12:
        phi = -phi
        phi -= phi[src].min()

    # Finally ensure non-negativity (guard against tiny numerical negatives)
    phi -= phi.min()
    return phi

# ------------------------------------------------------------
# Heat-method geodesics (tet/hypervolume)
# ------------------------------------------------------------
def heat_distance_tet(V, T, sources, t=None):
    """
    Heat-method geodesic distance in a tetrahedral mesh from 'sources' (vertex ids).
    Returns: d (n,) distances.
    """
    n = V.shape[0]
    if igl is None:
        # Fallback: graph Dijkstra using vertex adjacency from tets
        import heapq
        # build adjacency with edge length weights
        I, J = [], []
        for a,b,c,d in T:
            tet_edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
            for (p,q) in tet_edges:
                I.extend([p,q]); J.extend([q,p])
        I = np.array(I); J = np.array(J)
        Lw = np.linalg.norm(V[I]-V[J], axis=1)
        G = sp.coo_matrix((Lw, (I,J)), shape=(n,n)).tocsr()
        INF = 1e30
        d = np.full(n, INF)
        hq = []
        for s in np.atleast_1d(sources):
            d[s] = 0.0
            heapq.heappush(hq, (0.0, s))
        while hq:
            dist_u, u = heapq.heappop(hq)
            if dist_u != d[u]: continue
            for v in G[u].indices:
                w = G[u, v]
                nd = dist_u + w
                if nd < d[v]:
                    d[v] = nd
                    heapq.heappush(hq, (nd, v))
        d -= d.min()
        return d

    # libigl volumetric operators
    L = igl.cotmatrix(V, T)         # negative semi-definite volumetric Laplacian
    vol = _tet_volumes(V, T)
    M = _lumped_mass_from_areas(V.shape[0], T, vol)  # volumetric lumped mass

    # characteristic step t ~ h^2 with h = mean edge length
    if t is None:
        # collect all tet edges
        edges = set()
        for a,b,c,d in T:
            edges.update([(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)])
        edges = np.array(list({(min(i,j), max(i,j)) for (i,j) in edges}))
        h = np.linalg.norm(V[edges[:,0]] - V[edges[:,1]], axis=1).mean()
        t = h**2

    # Heat solve
    Aheat = (M - t * L)
    solve_heat = _safe_factor(Aheat)

    delta = np.zeros(n)
    src = np.atleast_1d(sources).astype(int)
    delta[src] = 1.0 / len(src)
    rhs_u = M @ delta
    u = solve_heat(rhs_u)

    # Gradient per tet (3m vector), normalize
    Gop = igl.grad(V, T)            # (3m x n)
    gu = (Gop @ u).reshape(-1, 3)   # per-tet gradient
    norms = np.linalg.norm(gu, axis=1, keepdims=True) + 1e-15
    X = -gu / norms

    # --- Step 3: divergence and Poisson solve ---
    # X is (mt,3); weight by tet volume, then flatten to (3*mt,)
    Xw = (X * vol[:, None]).reshape(-1)  # (3m,)
    div = - (Gop.T @ Xw)  # (n,)

    K = (-L)  # SPD
    solve_poisson = _safe_factor(K)
    phi = solve_poisson(div)

    # Shift so sources are at ~0
    src = np.atleast_1d(sources).astype(int)
    phi -= phi[src].min()

    # If everything is <= 0 (0 at sources, negative elsewhere), flip sign
    if np.max(phi) <= 1e-12:
        phi = -phi
        phi -= phi[src].min()

    # Finally ensure non-negativity (guard against tiny numerical negatives)
    phi -= phi.min()
    return phi

# ------------------------------------------------------------
# Farthest-Point Sampling using a distance oracle (surface or tet)
# ------------------------------------------------------------
def fps_with_distance(n_vertices, dist_from_sources_fn, k, start=None):
    """
    Generic FPS:
    - n_vertices: number of vertices in the mesh
    - dist_from_sources_fn(sources)->(n,) returns distance field
    - k: number of seeds
    - start: optional starting vertex id
    Returns: seeds (k,) array of vertex ids, min_dist (n,) final distances to nearest seed
    """
    if start is None:
        start = 0
    seeds = [int(start)]
    d = dist_from_sources_fn(seeds)
    # guard
    if not np.all(np.isfinite(d)):
        d = np.nan_to_num(d, nan=0.0, posinf=np.max(d[np.isfinite(d)]) if np.any(np.isfinite(d)) else 0.0)

    for _ in range(1, k):
        i = int(np.argmax(d))
        seeds.append(i)
        d_new = dist_from_sources_fn([i])
        if not np.all(np.isfinite(d_new)):
            d_new = np.nan_to_num(d_new, nan=np.max(d[np.isfinite(d)]) if np.any(np.isfinite(d)) else 0.0)
        d = np.minimum(d, d_new)
    return np.array(seeds, dtype=int), d

# ------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------
def surface_seeds_heat(V, F, k, start=None, t=None):
    n = V.shape[0]
    if start is None:
        start = 0
    # one warm-up step to move to a far vertex
    d0 = heat_distance_surface(V, F, [start], t=t)
    start = int(np.argmax(d0))
    return fps_with_distance(
        n_vertices=n,
        dist_from_sources_fn=lambda S: heat_distance_surface(V, F, S, t=t),
        k=k,
        start=start
    )

def tet_seeds_heat(V, T, k, start=None, t=None):
    return fps_with_distance(
        n_vertices=V.shape[0],
        dist_from_sources_fn=lambda S: heat_distance_tet(V, T, S, t=t),
        k=k,
        start=start if start is not None else 0
    )

def geodesic_labels_surface_from_seeds(V, F, seeds, t=None):
    seeds = np.asarray(seeds, int).ravel()
    D = np.column_stack([heat_distance_surface(V, F, [int(s)], t=t)  # NOTE: [s], not seeds
                         for s in seeds])  # D: (n, k)
    # (optional) enforce non-negativity per column
    D -= D.min(axis=0, keepdims=True)
    labels = np.argmin(D, axis=1)          # labels in [0..k-1], refer to seeds[labels[i]]
    return labels, D

def geodesic_labels_tet_from_seeds(V, T, seeds, t=None):
    """
    Per-vertex labels on a tetrahedral mesh using the heat method.
    seeds : 1D array of vertex IDs (length k)
    Returns:
      labels : (n,) int in [0..k-1], labels[i] refers to seeds[labels[i]]
      D      : (n,k) distances; column j is distance to seeds[j]
    """
    seeds = np.asarray(seeds, int).ravel()
    n, k = V.shape[0], len(seeds)
    # one heat-distance solve per seed (reuses factorizations inside heat_distance_tet)
    D = np.column_stack([heat_distance_tet(V, T, [int(s)], t=t) for s in seeds])
    # make each column non-negative with 0 near the source
    D -= D.min(axis=0, keepdims=True)
    labels = np.argmin(D, axis=1)
    return labels, D

def tet_labels_from_vertex_labels(T, vertex_labels, D=None, rule="mean"):
    """
    T: (m,4) tets
    vertex_labels: (n,) per-vertex labels in [0..k-1]
    D: optional (n,k) distances (needed for rule='mean' or 'min')
    rule: 'mode' | 'mean' | 'min'
      - 'mode': most frequent vertex label in the tet
      - 'mean': argmin_j mean(D[verts, j])  (requires D)
      - 'min' : argmin_j min(D[verts, j])   (requires D)
    """
    import numpy as np
    m = T.shape[0]
    if rule == "mode":
        from scipy.stats import mode
        lab = vertex_labels[T]               # (m,4)
        return mode(lab, axis=1, keepdims=False).mode
    else:
        assert D is not None, "D (n,k) distances required for rule='mean' or 'min'"
        k = D.shape[1]
        tet_vals = D[T]                      # (m,4,k)
        if rule == "mean":
            scores = tet_vals.mean(axis=1)   # (m,k)
        elif rule == "min":
            scores = tet_vals.min(axis=1)    # (m,k)
        else:
            raise ValueError("rule must be 'mode', 'mean', or 'min'")
        return np.argmin(scores, axis=1)

def load_medit_mesh(path):
    """
    Minimal MEDIT .mesh parser for vertices + tetrahedra.
    Returns V (n,3) float64, T (m,4) int32 (0-based)
    """
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]

    i = 0
    V = None
    T = None
    while i < len(lines):
        token = lines[i]
        if token.lower() == 'vertices':
            i += 1
            n = int(lines[i]); i += 1
            V = np.zeros((n, 3), dtype=float)
            # Each line: x y z region
            for k in range(n):
                parts = lines[i+k].split()
                V[k] = [float(parts[0]), float(parts[1]), float(parts[2])]
            i += n
        elif token.lower() == 'tetrahedra':
            i += 1
            m = int(lines[i]); i += 1
            T = np.zeros((m, 4), dtype=np.int32)
            # Each line: v1 v2 v3 v4 region  (1-based indices in MEDIT)
            for k in range(m):
                parts = lines[i+k].split()
                T[k] = [int(parts[0])-1, int(parts[1])-1, int(parts[2])-1, int(parts[3])-1]
            i += m
        else:
            i += 1
    if V is None or T is None:
        raise ValueError("Could not find Vertices and Tetrahedra sections in the .mesh file.")
    return V, T

## partitioning examples


def run_partitioning_example(k =2):
    # Load the Stanford bunny mesh (uploaded .obj)
    bunny = trimesh.load("data/bunny.obj")

    # Extract vertices and faces
    V = np.array(bunny.vertices)
    F = np.array(bunny.faces)

    # Sample seeds on surface (20 random vertices)
    seeds_ids, dist_to_nearest = surface_seeds_heat(V, F, k)  # vertex IDs for well-spaced seeds
    seeds = V[seeds_ids]
    # Compute Euclidean Voronoi partition by nearest seed (restricted to surface vertices)
    labels, D = geodesic_labels_surface_from_seeds(V, F, seeds_ids)

    # print("seeds", seeds)
    # print("finite ratio:", np.mean(np.isfinite(D)))
    # print("d min/max:", float(np.nanmin(D)), float(np.nanmax(D)))

    # --- 2D visualization ---
    plt.figure(figsize=(6,6))
    plt.scatter(V[:,0], V[:,1], c=labels, cmap="tab20", s=1)
    plt.scatter(seeds[:,0], seeds[:,1], c="red", s=30, marker="x")
    plt.title("2D Voronoi partition (projection of Stanford Bunny XY-plane)")
    plt.axis("equal")
    # plt.savefig("/mnt/data/bunny_voronoi_2d.png", dpi=300)
    plt.show()
    plt.close()

    # --- 3D visualization (approximate by coloring surface vertices by Voronoi label) ---
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(V[:,0], V[:,1], V[:,2], c=labels, cmap="tab20", s=1)
    ax.scatter(seeds[:,0], seeds[:,1], seeds[:,2], c="red", s=40, marker="x")
    ax.set_title("3D Voronoi partition (Stanford Bunny, Euclidean restricted to surface)")
    ax.view_init(elev=20, azim=30)
    # plt.savefig("/mnt/data/bunny_voronoi_3d.png", dpi=300)
    plt.show()
    plt.close()


    # --------- Load user mesh ----------
    mesh_path = "data/bar.mesh"
    V, T = load_medit_mesh(mesh_path)

    # --------- Seeds (pick K random tetrahedra barycenters as seeds) ----------
    rng = np.random.default_rng(42)
    tet_bary = V[T].mean(axis=1)        # (m,3)

    # 1) Get seeds as VERTEX IDs (k,)
    seed_vert, _ = tet_seeds_heat(V, T, k)   # returns vertex IDs

    # 2) Per-vertex geodesic distances/labels
    vlabels, D = geodesic_labels_tet_from_seeds(V, T, seed_vert)  # vlabels: (n,), D: (n,k)

    # 3) Convert to per-tet labels (match tet_bary size m)
    tlabels = tet_labels_from_vertex_labels(T, vlabels, D=D, rule="mean")  # (m,)
    #-- Save artifacts ----------
    # 1) Numpy dump (barycenters + labels) for reproducibility
    # np.savez("/mnt/data/bar_volumetric_voronoi_labels.npz", tet_bary=tet_bary, labels=labels, seeds=seeds, T=T, V=V)

    # 2) Try to export a VTK/VTU (if meshio is available) with cell data 'voronoi_label'
    vtk_path = None
    try:
        import meshio
        cells = [("tetra", T.astype(np.int32))]
        cell_data = {"voronoi_label": [labels.astype(np.int32)]}
        mesh = meshio.Mesh(points=V, cells=cells, cell_data=cell_data)
        vtk_path = "/mnt/data/bar_volumetric_voronoi.vtu"
        mesh.write(vtk_path)
    except Exception as e:
        vtk_path = None

    # 3) Simple 3D scatter of barycenters colored by label (one figure, default colormap)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    # 4) Plot barycenters colored by their tet label
    ax.scatter(tet_bary[:,0], tet_bary[:,1], tet_bary[:,2], c=tlabels, s=6)
    # (optional) plot seed points
    seed_pts = V[seed_vert]
    ax.scatter(seed_pts[:,0], seed_pts[:,1], seed_pts[:,2], marker='x', s=50)

    ax.set_title("Volumetric Voronoi (by tet barycenters)")
    ax.set_box_aspect([1,1,1])
    # plt.savefig("/mnt/data/bar_volumetric_voronoi_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    output = {
        "scatter_png": "/mnt/data/bar_volumetric_voronoi_scatter.png",
        "labels_npz": "/mnt/data/bar_volumetric_voronoi_labels.npz",
        "vtu": vtk_path if vtk_path else "VTU export unavailable (meshio not installed)"
    }

run_partitioning_example(k=20)