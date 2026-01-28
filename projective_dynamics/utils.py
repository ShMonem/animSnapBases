# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import os
import gdist  # for geodesics on tri mesh
# import skfmm  # for geodesics on tet mesh
import numpy as np
import polyscope as ps

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


def compute_surface_geodesics(vertices, faces, source_indices):
    # vertices: (n, 3) float64
    # faces: (m, 3) int32 (indices into vertices)
    # source_indices: 1D array of vertex indices where geodesics start

    distances = gdist.compute_gdist(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(source_indices, dtype=np.int32)
    )
    return distances  # shape (n,)

# Create a signed distance grid or binary mask
def compute_tet_geodesics(grid, source_coords):
    """
    grid: 3D binary volume (1: inside, 0: outside)
    source_coords: coordinates of the geodesic source point in voxel indices
    """
    phi = np.ones_like(grid, dtype=float)
    phi[source_coords] = -1  # set source
    phi = np.ma.MaskedArray(phi, mask=(grid == 0))  # mask outside

    dist = skfmm.distance(phi)
    return dist  # same shape as grid

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

def visualize_samples(V, F=None, T=None, samples=None, show_wireframe=False, point_radius=0.006):
    """
    Visualize mesh (tri or tet) and highlight sampled vertices using Polyscope.

    Parameters
    ----------
    V : (N, 3) array
        Vertex positions
    F : (M, 3) array or None
        Triangle faces (for surface meshes)
    T : (K, 4) array or None
        Tetrahedral cells (for volumetric meshes)
    samples : list[int] or np.ndarray
        Indices of sampled vertices
    show_wireframe : bool
        If True, show mesh in wireframe mode
    point_radius : float
        Display size of sampled points
    """

    ps.init()
    ps.set_up_dir("z_up")

    # Register the mesh (surface or volume)
    if F is not None:
        mesh = ps.register_surface_mesh("Mesh Surface", V, F, smooth_shade=True)
    elif T is not None:
        mesh = ps.register_volume_mesh("Tetrahedral Mesh", V, T)
    else:
        raise ValueError("You must provide either F (triangles) or T (tets).")

    # Show mesh in wireframe if requested
    if show_wireframe:
        mesh.set_edge_color((0.2, 0.2, 0.2))
        mesh.set_transparency(0.7)
        mesh.add_scalar_quantity("wireframe", np.zeros(len(V)), enabled=False)

    # Highlight sampled vertices
    if samples is not None and len(samples) > 0:
        samples = np.asarray(samples, dtype=int)
        sampled_positions = V[samples]
        ps.register_point_cloud("Sampled Vertices", sampled_positions, radius=point_radius, color=(1, 0, 0))
        print(f"✅ Displaying {len(samples)} sampled vertices out of {len(V)} total.")


def compute_face_normals(V, F):
    """
    Compute per-face normals (unnormalized, area-weighted).
    """
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # Cross product gives area-weighted normal
    face_normals = np.cross(v1 - v0, v2 - v0)
    return face_normals

def compute_vertex_normals(V, F):
    """
    Compute per-vertex normals as area-weighted averages of adjacent face normals.
    """
    n_verts = V.shape[0]
    v_normals = np.zeros((n_verts, 3), dtype=float)

    face_normals = compute_face_normals(V, F)

    for i in range(3):
        np.add.at(v_normals, F[:, i], face_normals)

    # Normalize
    norms = np.linalg.norm(v_normals, axis=1)
    nonzero = norms > 1e-12
    v_normals[nonzero] /= norms[nonzero][:, None]

    return v_normals

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


def rotate_mesh_once_x(V, angle=np.pi / 2):
    """
    Rotate mesh once around X axis by given angle.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ])

    # rotate around mesh center
    center = V.mean(axis=0)
    V_rot = (V - center) @ R.T + center

    return V_rot