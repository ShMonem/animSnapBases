# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from utils.process import veclen, normalized


def compute_tetMasses(vertexMasses, tetrahedrons, tetSize, auxiliarySize):
    assert auxiliarySize == 3
    masses =np.zeros(tetSize * auxiliarySize)
    for t in range(tetrahedrons.shape[0]):
        weights = np.array([vertexMasses[tetrahedrons[t, 0]], vertexMasses[tetrahedrons[t, 1]],
                  vertexMasses[tetrahedrons[t, 2]], vertexMasses[tetrahedrons[t, 3]]])
        weight = weights.sum()

        for k in range(auxiliarySize):
            masses[t * auxiliarySize + k] = weight

    return masses


def compute_edgeMasses(vertexMasses, edges, edgeSize, auxiliarySize):
    masses = np.zeros(edgeSize * auxiliarySize)

    for e in range(edges.shape[0]):
        v1, v2 = edges[e]
        weight = vertexMasses[v1] + vertexMasses[v2]
        if weight == 0:
            print(f"Warning: zero weight on edge {e} ({v1}, {v2})")

        for k in range(auxiliarySize):
            masses[e * auxiliarySize + k] = weight

    return masses


def compute_lumped_mass_matrix(V, T, density=1.0):
    from scipy.sparse import coo_matrix
    def volume_of_tet(v0, v1, v2, v3):
        return abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0

    n = V.shape[0]
    mass_per_vertex = np.zeros(n)

    for tet in T:
        v0, v1, v2, v3 = V[tet]
        vol = volume_of_tet(v0, v1, v2, v3)
        lumped_mass = density * vol / 4.0  # equally divided over 4 verts
        for i in tet:
            mass_per_vertex[i] += lumped_mass
        # Normalize: total mass becomes 1
    total_mass = mass_per_vertex.sum()
    if total_mass > 0:
        mass_per_vertex /= total_mass
    return coo_matrix((mass_per_vertex, (range(n), range(n))), shape=(n, n))


def compute_triMasses(vertexMasses, triangles, triSize, auxiliarySize):

    assert auxiliarySize == 2
    masses =np.zeros(triSize * auxiliarySize)
    for t in range(triangles.shape[0]):
        weights = np.array([vertexMasses[triangles[t, 0]], vertexMasses[triangles[t, 1]],
                  vertexMasses[triangles[t, 2]]])
        weight = weights.sum()
        if weight == 0:
            print("zero weight")

        for k in range(auxiliarySize):
            masses[t * auxiliarySize + k] = weight

    return masses
'''
The following functions are borrowed from: https://github.com/tneumann/splocs
Copyright (c) 2013 Thomas Neumann
'''
def compute_mesh_laplacian(verts, tris):
    """
    computes a sparse matrix representing the discretized laplace-beltrami operator of the mesh
    given by n vertex positions ("verts") and a m triangles ("tris")

    verts: (n, 3) array (float)
    tris: (m, 3) array (int) - indices into the verts array

    computes the conformal weights ("cotangent weights") for the mesh, ie:
    w_ij = - .5 * (cot \alpha + cot \beta)

    See:
        Olga Sorkine, "Laplacian Mesh Processing"
        and for theoretical comparison of different discretizations, see
        Max Wardetzky et al., "Discrete Laplace operators: No free lunch"

    returns matrix L that computes the laplacian coordinates, e.g. L * x = delta
    """
    n = len(verts)
    W_ij = np.empty(0)
    I = np.empty(0, np.int32)
    J = np.empty(0, np.int32)
    for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:  # for edge i2 --> i3 facing vertex i1
        vi1 = tris[:, i1]  # vertex index of i1
        vi2 = tris[:, i2]
        vi3 = tris[:, i3]
        # vertex vi1 faces the edge between vi2--vi3
        # compute the angle at v1
        # add cotangent angle at v1 to opposite edge v2--v3
        # the cotangent weights are symmetric
        u = verts[vi2] - verts[vi1]
        v = verts[vi3] - verts[vi1]
        cotan = (u * v).sum(axis=1) / veclen(np.cross(u, v))
        W_ij = np.append(W_ij, 0.5 * cotan)
        I = np.append(I, vi2)
        J = np.append(J, vi3)
        W_ij = np.append(W_ij, 0.5 * cotan)
        I = np.append(I, vi3)
        J = np.append(J, vi2)
    L = sparse.csr_matrix((W_ij, (I, J)), shape=(n, n))
    # compute diagonal entries
    L = L - sparse.spdiags(L * np.ones(n), 0, n, n)
    L = L.tocsr()
    # area matrix
    e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    n = np.cross(e1, e2)
    triangle_area = .5 * veclen(n)
    # compute per-vertex area
    vertex_area = np.zeros(len(verts))
    ta3 = triangle_area / 3
    for i in range(tris.shape[1]):
        bc = np.bincount(tris[:, i].astype(int), ta3)
        vertex_area[:len(bc)] += bc
    VA = sparse.spdiags(vertex_area, 0, len(verts), len(verts))
    return L, VA


class GeodesicDistanceComputation(object):
    """
    Computation of geodesic distances on triangle meshes using the heat method from the impressive paper

        Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow
        Keenan Crane, Clarisse Weischedel, Max Wardetzky
        ACM Transactions on Graphics (SIGGRAPH 2013)

    Example usage:
        >>> compute_distance = GeodesicDistanceComputation(vertices, triangles)
        >>> distance_of_each_vertex_to_vertex_0 = compute_distance(0)

    """

    def __init__(self, verts, tris, m=10.0):
        self._verts = verts
        self._tris = tris
        # precompute some stuff needed later on
        e01 = verts[tris[:, 1]] - verts[tris[:, 0]]
        e12 = verts[tris[:, 2]] - verts[tris[:, 1]]
        e20 = verts[tris[:, 0]] - verts[tris[:, 2]]
        self._triangle_area = .5 * veclen(np.cross(e01, e12))
        unit_normal = normalized(np.cross(normalized(e01), normalized(e12)))
        self._unit_normal_cross_e01 = np.cross(unit_normal, e01)
        self._unit_normal_cross_e12 = np.cross(unit_normal, e12)
        self._unit_normal_cross_e20 = np.cross(unit_normal, e20)
        # parameters for heat method
        h = np.mean(list(map(veclen, [e01, e12, e20])))
        t = m * h ** 2
        # pre-factorize poisson systems
        Lc, A = compute_mesh_laplacian(verts, tris)
        self._factored_AtLc = splu((A - t * Lc).tocsc()).solve
        self._factored_L = splu(Lc.tocsc()).solve

    def __call__(self, idx):
        """
        computes geodesic distances to all vertices in the mesh
        idx can be either an integer (single vertex index) or a list of vertex indices
        or an array of bools of length n (with n the number of vertices in the mesh)
        """
        u0 = np.zeros(len(self._verts))
        u0[idx] = 1.0
        # heat method, step 1
        u = self._factored_AtLc(u0).ravel()
        # heat method step 2
        grad_u = 1 / (2 * self._triangle_area)[:, np.newaxis] * (
                self._unit_normal_cross_e01 * u[self._tris[:, 2]][:, np.newaxis]
                + self._unit_normal_cross_e12 * u[self._tris[:, 0]][:, np.newaxis]
                + self._unit_normal_cross_e20 * u[self._tris[:, 1]][:, np.newaxis]
        )
        X = - grad_u / veclen(grad_u)[:, np.newaxis]
        # heat method step 3
        div_Xs = np.zeros(len(self._verts))
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:  # for edge i2 --> i3 facing vertex i1
            vi1, vi2, vi3 = self._tris[:, i1], self._tris[:, i2], self._tris[:, i3]
            e1 = self._verts[vi2] - self._verts[vi1]
            e2 = self._verts[vi3] - self._verts[vi1]
            e_opp = self._verts[vi3] - self._verts[vi2]
            cot1 = 1 / np.tan(np.arccos(
                (normalized(-e2) * normalized(-e_opp)).sum(axis=1)))
            cot2 = 1 / np.tan(np.arccos(
                (normalized(-e1) * normalized(e_opp)).sum(axis=1)))
            div_Xs += np.bincount(
                vi1.astype(int),
                0.5 * (cot1 * (e1 * X).sum(axis=1) + cot2 * (e2 * X).sum(axis=1)),
                minlength=len(self._verts))
        phi = self._factored_L(div_Xs).ravel()
        phi -= phi.min()
        # print(phi.max())
        return phi

def get_triangles_per_vert(vertex_indices, tris):
    # Convert the list of vertex indices to a set for faster lookup
    vertex_indices_set = set(vertex_indices)

    # List to hold the tetrahedrons that include any of the specified vertices
    matching_tri_indices = []
    count = 0
    # Iterate over each tetrahedron
    for index, tri in enumerate(tris):
        # Check if any vertex of the tetrahedron is in the vertex_indices_set
        if any(vertex in vertex_indices_set for vertex in tri):
            matching_tri_indices.append(index)

    return matching_tri_indices

def get_tetrahedrons_per_vert(vertex_indices, tets):
    # Convert the list of vertex indices to a set for faster lookup
    vertex_indices_set = set(vertex_indices)

    # List to hold the tetrahedrons that include any of the specified vertices
    matching_tet_indices = []
    count = 0
    # Iterate over each tetrahedron
    for index, tet in enumerate(tets):
        # Check if any vertex of the tetrahedron is in the vertex_indices_set
        if any(vertex in vertex_indices_set for vertex in tet):
            matching_tet_indices.append(index)

    return matching_tet_indices

def get_vert_star_per_vert(vertex_index, faces):
    neighbors = set()
    for face in faces:
        if vertex_index in face:
            # Add all vertices of the face to the neighbors set but remove the original vertex
            neighbors.update(face)
    #neighbors.discard(vertex_index)  # Remove the original vertex from the set of neighbors
    return list(neighbors)

def get_edges_per_vert(vertex_indices, edges):
    vertex_indices_set = set(vertex_indices)
    matching_edge_indices = []

    for index, edge in enumerate(edges):
        if edge[0] in vertex_indices_set or edge[1] in vertex_indices_set:
            matching_edge_indices.append(index)

    return matching_edge_indices


def compute_edge_incidence_matrix_on_tets(tets):
    """
    Computes the edge incidence matrix for a tetrahedral mesh.

    Parameters:
    - tets: np.ndarray, array of tetrahedral indices.

    Returns:
    - edges: np.ndarray, array of edges (unique pairs of vertices) with consistent orientation.
    """
    edges_set = set()

    for tet in tets:
        # Extract edges from each tetrahedron
        tet_edges = [
            (tet[0], tet[1]),
            (tet[0], tet[2]),
            (tet[0], tet[3]),
            (tet[1], tet[2]),
            (tet[1], tet[3]),
            (tet[2], tet[3])
        ]

        # Add edges to the set with consistent orientation (smallest index first)
        for edge in tet_edges:
            edges_set.add(tuple(sorted(edge)))

    # Convert the set to a sorted array for consistent ordering
    edges = np.array(sorted(edges_set))
    return edges

def compute_edge_incidence_matrix_on_tris(tris):
    """
    Computes the edge incidence matrix for a tetrahedral mesh.

    Parameters:
    - tets: np.ndarray, array of tetrahedral indices.

    Returns:
    - edges: np.ndarray, array of edges (unique pairs of vertices) with consistent orientation.
    """
    edges_set = set()

    for tri in tris:
        # Extract edges from each face
        tri_edges = [
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[2], tri[0])
        ]

        # Add edges to the set with consistent orientation (smallest index first)
        for edge in tri_edges:
            edges_set.add(tuple(sorted(edge)))

        # Convert the set to a sorted array for consistent ordering
    edges = np.array(sorted(edges_set))
    return edges


def extract_sub_vertices_and_edges(vertices, sub_edges):
    """
    Extracts the subset of vertices and remaps edges for a subset of edges.

    Parameters:
    - vertices: np.ndarray, array of vertex positions.
    - sub_edges: np.ndarray, (e, 2) array of edges (pairs of vertex indices).

    Returns:
    - sub_vertices: np.ndarray, array of positions for the subset of vertices.
    - remapped_edges: np.ndarray, edges remapped to the subset of vertices.
    """
    # Find unique vertices in the subset of edges
    unique_vertex_indices = np.unique(sub_edges)

    # Create a mapping from global indices to local indices
    index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(unique_vertex_indices)}

    # Remap edges to the local vertex indices
    remapped_edges = np.array([[index_map[edge[0]], index_map[edge[1]]] for edge in sub_edges])

    # Extract the subset of vertex positions
    sub_vertices = vertices[unique_vertex_indices]

    return sub_vertices, remapped_edges

def extract_sub_vertices_and_tet_edges(vertices, sub_edges):
    """
    Extracts the subset of vertices and remaps tetrahedral edges for a subset of edges.

    Parameters:
    - vertices: np.ndarray, array of vertex positions.
    - sub_edges: np.ndarray, (e, 2) array of edges (pairs of vertex indices).

    Returns:
    - sub_vertices: np.ndarray, array of positions for the subset of vertices.
    - remapped_edges: np.ndarray, edges remapped to the subset of vertices.
    """
    # Find unique vertices in the subset of edges
    unique_vertex_indices = np.unique(sub_edges)

    # Create a mapping from global indices to local indices
    index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(unique_vertex_indices)}

    # Remap edges to the local vertex indices
    remapped_edges = np.array([[index_map[edge[0]], index_map[edge[1]]] for edge in sub_edges])

    # Extract the subset of vertex positions
    sub_vertices = vertices[unique_vertex_indices]

    return sub_vertices, remapped_edges
