import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, vstack
from scipy.linalg import cholesky, solve, eigh, svd
try:
    import gdist
    USE_GDIST = True
except ImportError:
    USE_GDIST = False
    from pygeodesic import geodesic


from stvd import STVD, STVD_MODE_ST_DIJKSTRA, STVD_MODE_GRAPH_DIJKSTRA
import math
from utils import compute_surface_geodesics

BASE_FUNC_CUTOFF = 1e-5
USE_QUARTIC_POL = False  # or True for quartic RBFs


def check_matrix_health(matrix, name="matrix"):
    try:
        arr = np.asarray(matrix, dtype=np.float64)

        if np.isnan(arr).any():
            print(f"[Warning] {name} contains NaN values.")
        if np.isinf(arr).any():
            print(f"[Warning] {name} contains Inf values.")
        if np.abs(arr).max() > 1e10:
            print(f"[Warning] {name} contains very large values.")
        if np.abs(arr).min() < 1e-10 and not np.all(arr == 0):
            print(f"[Warning] {name} contains very small (nonzero) values.")

    except Exception as e:
        print(f"[Error] Could not process {name}: {e}")


def create_skinning_space(
    rest_positions,
    weights,
    for_constraints=None,
    lines_per_constraint=1,
    tris=None,
    tets=None,
    flat_space=False,
):
    """
    Equivalent to PD::createSkinningSpace from C++ HRPD code.
    Builds either the position subspace (U) or constraint projection subspace (V),
    depending on whether 'for_constraints' is provided.

    Parameters
    ----------
    rest_positions : (N, D) np.ndarray
        Rest-state vertex or auxiliary positions (usually 3D).
    weights : (N, G) np.ndarray
        Skinning weights matrix (each vertex/element's weight for G groups).
    for_constraints : list or None
        If provided, list of constraint objects (each has .main_vertex,
        .main_triangle, .main_tet attributes or similar getters).
        If None → positional subspace construction (U).
    lines_per_constraint : int
        Number of rows per constraint in rest_positions (auxiliary size).
    tris : (M, 3) np.ndarray or None
        Triangle connectivity (optional, only needed if constraints are triangle-based).
    tets : (L, 4) np.ndarray or None
        Tetrahedral connectivity (optional, only needed if constraints are tet-based).
    flat_space : bool
        If True, uses D-1 dimension (for 2D projections / flat embeddings).

    Returns
    -------
    skinning_space : (N, G*(D+1) + 1) np.ndarray
        The constructed skinning space matrix.
    """

    if np.isnan(weights).any() or np.isnan(rest_positions).any():
        print(" Warning: NaN detected in weights or rest positions.")

    k = weights.shape[1]
    num_rows = rest_positions.shape[0]
    dim = rest_positions.shape[1] - (1 if flat_space else 0)
    if for_constraints is not None:
        skinning_space = np.zeros((num_rows,(dim + 1)* k + 1), dtype=float)
    else:
        skinning_space = np.zeros((3*num_rows, 4 * k+1) , dtype=float)

    error = False

    for v in range(num_rows):
        for g in range(k):
            cur_weight = 0.0

            # -----------------------------
            #  Case 1: constraint-based mode (constraint projection space)
            # -----------------------------
            if for_constraints is not None:
                c = for_constraints[v // lines_per_constraint]
                cur_weight = weights[c.id, g]

            # -----------------------------
            #  Case 2: vertex-based mode (position subspace)
            # -----------------------------
            else:
                cur_weight = weights[v, g]

            if np.isnan(cur_weight):
                print(" Warning: NaN weight during skinning space construction.")
                error = True
                continue

            # fill coordinate terms
            for d in range(dim):
                skinning_space[v, g * (dim + 1) + d] = rest_positions[v, d] * cur_weight

            # fill weight-only term
            skinning_space[v, g * (dim + 1) + dim] = cur_weight

    if error:
        print(" Errors encountered during construction. Returning empty matrix.")
        return np.zeros((0, 0))

    # Add constant column of 1s (for affine term)
    skinning_space[:, -1] = 1.0

    if np.isnan(skinning_space).any():
        print(" Warning: NaN entries in constructed skinning space!")

    return skinning_space


class Sampler:
    def __init__(self, positions, tris=None, tets=None):
        self.stvd = STVD()
        self.n_vertices = 0
        self.vertices = None
        self.tris = None
        self.tets = None
        self.distances = None
        self.up_to_date = False
        self.vertices = positions
        self.tris = tris
        self.tets = tets
        self.n_vertices = positions.shape[0]
        self.stvd.init(positions, tris, tets)
        self.up_to_date = False

    def clear_sources(self):
        self.stvd.reset_sources()
        self.stvd.reset_distances()

    def add_source(self, index):
        self.stvd.add_source(index)
        self.up_to_date = False

    def compute_distances(self, max_dist=-1.0, accurate=False):
        mode = STVD_MODE_ST_DIJKSTRA if accurate else STVD_MODE_GRAPH_DIJKSTRA
        self.stvd.compute_distances(True, mode, max_dist)
        self.distances = self.stvd.get_distances()
        self.up_to_date = True

    def get_distance(self, idx):
        if not self.up_to_date:
            raise RuntimeError("Distances not up to date, call compute_distances first.")
        return self.distances[idx]

    # ---------------------------------------------------------
    def get_samples(self, num_samples):
        samples = []
        self.clear_sources()
        first_vert = np.random.randint(0, self.n_vertices)
        samples.append(first_vert)
        self.add_source(first_vert)
        self.add_samples(num_samples - 1, samples)
        return samples

    def add_samples(self, num_samples, samples):
        if num_samples <= 0:
            return
        for _ in range(num_samples):
            self.compute_distances()
            best_vert = np.argmax(self.distances)
            attempt = 0
            while best_vert in samples and attempt < self.n_vertices:
                self.distances[best_vert] = -np.inf
                best_vert = np.argmax(self.distances)
                attempt += 1
            if attempt >= self.n_vertices:
                print("Sampling terminated early.")
                return
            samples.append(best_vert)
            self.add_source(best_vert)

    def extend_samples(self, num_total_samples, current_samples):
        if num_total_samples <= len(current_samples):
            return
        self.clear_sources()
        for s in current_samples:
            self.add_source(s)
        self.add_samples(num_total_samples - len(current_samples), current_samples)

    # ---------------------------------------------------------
    def get_sample_diameter(self, samples):
        self.clear_sources()
        for v in samples:
            self.add_source(v)
        self.compute_distances()
        return np.max(self.distances)

    def get_radial_base_functions(
        self, samples, partition_of_one=True, r=1.0, eps=-1.0,
        num_small_samples=0, small_sample_radius=0.5
    ):
        if eps < 0:
            eps = math.sqrt(-math.log(BASE_FUNC_CUTOFF)) / r

        n_samples = len(samples)
        base_functions = np.zeros((self.n_vertices, n_samples))
        a = 1.0 / (r ** 4)
        b = -2.0 / (r ** 2)

        for i, s in enumerate(samples):
            if num_small_samples > 0 and i > n_samples - num_small_samples:
                r = small_sample_radius
                eps = math.sqrt(-math.log(BASE_FUNC_CUTOFF)) / r

            self.clear_sources()
            self.add_source(s)
            self.compute_distances(r, accurate=False)

            for v in range(self.n_vertices):
                d = self.get_distance(v)
                if d < 0:
                    val = 0
                elif USE_QUARTIC_POL:
                    if d >= r:
                        val = 0
                    else:
                        val = a * (d ** 4) + b * (d ** 2) + 1
                else:
                    val = math.exp(-(d * eps * d * eps))
                    if val < BASE_FUNC_CUTOFF:
                        val = 0
                base_functions[v, i] = val

        # normalize (partition of unity)
        if partition_of_one:
            row_sums = base_functions.sum(axis=1)
            for v in range(self.n_vertices):
                if row_sums[v] < 1e-6:
                    max_idx = np.argmax(base_functions[v, :])
                    base_functions[v, :] = 0
                    base_functions[v, max_idx] = 1.0
                else:
                    base_functions[v, :] /= row_sums[v]
        return base_functions


class PositionsSubspace:
    def __init__(self, vertices, faces=None, tets=None, num_samples=-1):
        self.U = None
        self.rest_pos = vertices
        self.faces = faces
        self.tets = tets
        self.sampler = Sampler(vertices, faces, tets)
        self.vert_samples = self.sampler.get_samples(num_samples)
        self.num_components = num_samples
        self.weights = None

    def compute_skinning_weights(self, r_factor=4.0):
        # Functions to compute LBS-subspace for positions
        def br_cubic(t, r):
            """Compactly supported cubic radial basis function."""
            t = np.clip(t, 0, r)
            return (1 - (t / r) ** 3) ** 3

        def quartic_rbf(dist, r):
            a = 1 / r ** 4
            b = -2 / r ** 2
            val = a * dist ** 4 + b * dist ** 2 + 1
            val[dist >= r] = 0
            return val

        """
        Compute subspace matrix U ∈ R^{n×4k} for position encoding.

        Parameters:
            vertices: (n, 3) array of vertex positions
            k: number of sample points (i.e. handles)
            r_factor: scaling for support radius r

        Returns:
            U: (n, 4k) sparse matrix
        """
        vertices = self.rest_pos
        n = vertices.shape[0]

        # === Step 1: Furthest Point Sampling (Euclidean approximation) ===
        samples = [np.random.randint(n)]
        for _ in range(1, self.num_components):
            dists = np.min(np.linalg.norm(vertices[samples][:, None] - vertices[None, :], axis=2), axis=0)
            next_sample = np.argmax(dists)
            samples.append(next_sample)

        samples = np.array(samples)

        # === Step 2: Compute RBF weights ===
        # Use gdist to compute distances from all samples
        geo_dists = np.zeros((vertices.shape[0], len(samples)))
        for i, s in enumerate(samples):
            geo_dists[:, i] = compute_surface_geodesics(vertices, self.faces, [s])

        r = r_factor * np.max(np.min(geo_dists, axis=1))

        weights = quartic_rbf(geo_dists, r)

        # Normalize weights so that they sum to 1 per vertex
        weight_sums = np.sum(weights, axis=1, keepdims=True) + 1e-8
        weights /=weight_sums

        # Handling Uncovered Vertices
        for i in range(n):
            if np.sum(weights[i, :]) < 1e-6:
                max_idx = np.argmin(geo_dists[i, :])
                weights[i, :] = 0
                weights[i, max_idx] = 1

        return weights

    def create_basis_via_skinning_weights(self):

        weights = self.compute_skinning_weights()

        n = self.rest_pos.shape[0]
        # === Step 3: Build U matrix ===
        row_idx = []
        col_idx = []
        data = []

        for i in range(n):  # for each vertex
            for j in range(self.num_components):
                w = weights[i, j]
                if w > 1e-6:
                    for d in range(3):  # x, y, z
                        row_idx.append(i * 3 + d)
                        col_idx.append(j * 4 + d)
                        data.append(w)
                    # Last DOF is uniform scale/shear control
                    row_idx.append(i * 3 + 0)
                    col_idx.append(j * 4 + 3)
                    data.append(w * self.rest_pos[i, 0])
                    row_idx.append(i * 3 + 1)
                    col_idx.append(j * 4 + 3)
                    data.append(w * self.rest_pos[i, 1])
                    row_idx.append(i * 3 + 2)
                    col_idx.append(j * 4 + 3)
                    data.append(w * self.rest_pos[i, 2])

        self.U = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n * 3, 4 * self.num_components)).tocsr()


class ConstraintsProjectionSubspace:
    def __init__(self, constraint_name, vertices, faces=None, tets=None, num_samples=-1):
        self.V = None
        self.vertices = vertices
        self.faces = faces
        self.tets = tets
        self.sampler = Sampler(vertices, faces, tets)
        self.vert_samples = self.sampler.get_samples(num_samples)
        self.num_components = num_samples
        self.constraint_name = constraint_name
        self.weights = None
        self.mass = None
        self.sampled_constraints_ids = []
        self.interpol_solver = {}
        self.alpha_points = []

    def get_sampled_constrained_elements(self):

        results = []
        def find_first_nighbours(elements):
            for v in self.vert_samples:
                inds = np.where(np.any(elements == v, axis=1))[0]
                if len(inds) == 0:
                    results.append(None)
                else:
                    idx = inds[0]
                    results.append(idx)
            return list(set(results))

        if self.constraint_name == "verts_bending":
             self.sampled_constraints_ids = self.vert_samples
        elif self.constraint_name == "edge_spring":
            self.sampled_constraints_ids = find_first_nighbours(self.edges)
        elif self.constraint_name == "tris_strain":
            self.sampled_constraints_ids = find_first_nighbours(self.faces)
        elif self.constraint_name in {"tets_strain", "tets_deformation_gradient"}:
            self.sampled_constraints_ids = find_first_nighbours(self.tets)
        else:
            raise  ValueError(f"Unknown constraint type {self.constraint_name} for LBE subspaces")

    def get_edge_weights_from_vertex_weights(self, vertex_weights):
        """
        Convert vertex-based weights to edge-based weights (mean per edge),
        Args:
            vertex_weights: (n_vertices, k) numpy array

        Returns:
            edge_weights: (n_edges, k) numpy array of edge-based weights
        """
        # Ensure edges exist
        if not hasattr(self, "edges") or self.edges is None:
            # Extract all edges (unordered pairs)
            edges = np.vstack([
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]]
            ])

            # Sort each edge so (i, j) and (j, i) are treated the same
            edges = np.sort(edges, axis=1)

            # Get unique edges
            self.edges = np.unique(edges, axis=0)

        n_edges = self.edges.shape[0]
        n_groups = vertex_weights.shape[1]
        edge_weights = np.zeros((n_edges, n_groups))

        for e in range(n_edges):
            # Average vertex weights for this edge (2 vertices)
            edge_weights[e, :] = np.mean(vertex_weights[self.edges[e, :], :], axis=0)

            # Normalize (partition of unity)
            row_sum = np.sum(edge_weights[e, :])
            if row_sum > 1e-8:
                edge_weights[e, :] /= row_sum
            else:
                # Fallback: assign all weight to the maximum group
                max_idx = np.argmax(edge_weights[e, :])
                edge_weights[e, :] = 0
                edge_weights[e, max_idx] = 1.0

        return edge_weights

    def get_triangle_weights_from_vertex_weights(self, vertex_weights):
        """
        Convert vertex-based weights to triangle-based weights (mean per triangle),
        matching PD::getTriangleWeightsFromVertexWeights.

        Args:
            vertex_weights: (n_vertices, k) numpy array
            tris: (n_triangles, 3) integer numpy array of vertex indices

        Returns:
            tri_weights: (n_triangles, k) numpy array of triangle-based weights
        """
        n_tris = self.faces.shape[0]
        n_groups = vertex_weights.shape[1]
        tri_weights = np.zeros((n_tris, n_groups))

        for t in range(n_tris):
            # Average vertex weights for this triangle
            tri_weights[t, :] = np.mean(vertex_weights[self.faces[t, :], :], axis=0)

            # Normalize (partition of unity)
            row_sum = np.sum(tri_weights[t, :])
            if row_sum > 1e-8:
                tri_weights[t, :] /= row_sum
            else:
                # fallback: assign all weight to max value group
                max_idx = np.argmax(tri_weights[t, :])
                tri_weights[t, :] = 0
                tri_weights[t, max_idx] = 1.0

        return tri_weights

    def get_tet_weights_from_vertex_weights(self, vertex_weights):
        """
        Convert vertex-based RBF weights to tetrahedron-based weights (mean per tet),
        matching PD::getTetWeightsFromVertexWeights.

        Args:
            vertex_weights: (n_vertices, k) numpy array
            tets: (n_tets, 4) integer numpy array of vertex indices

        Returns:
            tet_weights: (n_tets, k) numpy array of tetrahedral weights
        """
        n_tets = self.tets.shape[0]
        n_groups = vertex_weights.shape[1]
        tet_weights = np.zeros((n_tets, n_groups))

        for t in range(n_tets):
            # Average vertex weights for this tetrahedron
            tet_weights[t, :] = np.mean(vertex_weights[self.tets[t, :], :], axis=0)

            # Normalize (partition of unity)
            row_sum = np.sum(tet_weights[t, :])
            if row_sum > 1e-8:
                tet_weights[t, :] /= row_sum
            else:
                # fallback: assign all weight to the strongest RBF
                max_idx = np.argmax(tet_weights[t, :])
                tet_weights[t, :] = 0.0
                tet_weights[t, max_idx] = 1.0

        return tet_weights

    def compute_skinning_weights(self, r_multiplier=1.1):
        max_distance= self.sampler.get_sample_diameter(self.vert_samples)

        weights = self.sampler.get_radial_base_functions(self.vert_samples, r = max_distance*r_multiplier, partition_of_one=True)

        # r her
        # def quartic_rbf(dist, r):
        #     a = 1 / r ** 4
        #     b = -2 / r ** 2
        #     val = a * dist ** 4 + b * dist ** 2 + 1
        #     val[dist >= r] = 0
        #     return val
        #
        # """
        # Compute subspace matrix U ∈ R^{n×4k} for position encoding.
        #
        # Parameters:
        #     vertices: (n, 3) array of vertex positions
        #     k: number of sample points (i.e. handles)
        #     r_factor: scaling for support radius r
        #
        # Returns:
        #     U: (n, 4k) sparse matrix
        # """
        # vertices = self.vertices
        # n = vertices.shape[0]
        #
        # # === Step 1: Furthest Point Sampling (Euclidean approximation) ===
        # samples = [np.random.randint(n)]
        # for _ in range(1, self.num_components):
        #     dists = np.min(np.linalg.norm(vertices[samples][:, None] - vertices[None, :], axis=2), axis=0)
        #     next_sample = np.argmax(dists)
        #     samples.append(next_sample)
        #
        # samples = np.array(samples)
        #
        # # === Step 2: Compute RBF weights ===
        # # Use gdist to compute distances from all samples
        # geo_dists = np.zeros((vertices.shape[0], len(samples)))
        # for i, s in enumerate(samples):
        #     geo_dists[:, i] = compute_surface_geodesics(vertices, self.faces, [s])
        #
        # r = r_multiplier * np.max(np.min(geo_dists, axis=1))
        #
        # weights = quartic_rbf(geo_dists, r)
        #
        # # Normalize weights so that they sum to 1 per vertex
        # weight_sums = np.sum(weights, axis=1, keepdims=True) + 1e-8
        # weights /=weight_sums
        #
        # # Handling Uncovered Vertices
        # for i in range(n):
        #     if np.sum(weights[i, :]) < 1e-6:
        #         max_idx = np.argmin(geo_dists[i, :])
        #         weights[i, :] = 0
        #         weights[i, max_idx] = 1

        check_matrix_health(weights, "weights")

        if self.constraint_name == "verts_bending":
             self.weights = weights
        elif self.constraint_name == "edge_spring":
             self.weights = self.get_edge_weights_from_vertex_weights(weights)
        elif self.constraint_name == "tris_strain":
            self.weights = self.get_triangle_weights_from_vertex_weights(weights)
        elif self.constraint_name in {"tets_strain", "tets_deformation_gradient"}:
            self.weights = self.get_tet_weights_from_vertex_weights(weights)
        else:
            raise  ValueError(f"Unknown constraint type {self.constraint_name} for LBE subspaces")

    def compute_constraint_mass_matrix(self, group_name, constraints, vertex_masses, aux_size):
        """
        Compute mass matrix and its diagonal for the constraint projection auxiliaries.

        Args:
            constraints: list of dicts, each with keys like {"vertex": i} or {"triangle": t} or {"tet": T}
            vertex_masses: (n,) array of per-vertex masses
            tris: (m, 3) array of triangle indices
            tets: (p, 4) array of tetrahedron indices
            aux_size: number of auxiliary variables per constraint (same as in C++)
            no_masses: if True, set all masses to 1
        Returns:
            M_diag: (len(constraints)*aux_size,) numpy array of diagonal mass entries
        """
        num_constraints = len(constraints)
        M_diag = np.zeros(num_constraints * aux_size)

        if group_name == "verts_bending":
            self.mass = vertex_masses
        else:
            for i, c in enumerate(constraints):
                w = np.sum(vertex_masses[c.indices])

                M_diag[i * aux_size:(i + 1) * aux_size] = w

            self.mass = M_diag

    def snapshot_pca(self, Y, specify_verts):
        """
        Perform mass-weighted PCA (Method of Snapshots) equivalent to PD::snapshotPCA.

        Args:
            Y: (m, s) snapshot matrix (each column = constraint-space feature)
            masses_diag: (m,) mass weights vector
            size: number of principal components to keep

        Returns:
            base: (m, size+1) reduced PCA basis
        """
        # Step 1: remove column-wise mean
        Y_centered = Y - Y.mean(axis=0, keepdims=True)

        # Step 2: compute A = Y^T M Y
        if specify_verts:
            M = diags(self.mass[specify_verts])
        else:
            M = diags(self.mass)
        A = Y_centered.T @ (M @ Y_centered)

        # Step 3: eigendecomposition
        vals, vecs = eigh(A)

        # Sort in descending order
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]

        # # Step 4: keep top 'size' modes
        # vals, vecs = vals[:self.num_components], vecs[:, :self.num_components]

        # Step 5: compute PCA basis = Y * u / sqrt(lambda)
        base = Y_centered @ (vecs / np.sqrt(vals + 1e-12))

        # Step 6: add global translation vector
        base[:, -1] = 1.0

        self.V = base

    def create_skinning_space_constraints(self, rest_state_aux, skinning_weights, constraints, aux_size):

        """
        Equivalent of PD::createSkinningSpace with forConstraints != nullptr
        Builds the constraint-space skinning matrix Y.
        """
        num_basis = skinning_weights.shape[1]
        num_rows, dim = rest_state_aux.shape
        Y = np.zeros((num_rows, num_basis * (dim + 1) + 1))

        for i, c in enumerate(constraints):
            for g in range(num_basis):
                # Find which element the constraint refers to
                cur_weight = skinning_weights[c.id, g]
                for v in range(aux_size):
                    for d in range(dim):
                        Y[i*aux_size+ v, g * (dim + 1) + d] = rest_state_aux[i*aux_size+ v, d] * cur_weight
                    Y[i*aux_size+ v, g * (dim + 1) + dim] = cur_weight

        # Final column = constant 1
        Y[:, -1] = 1.0
        return Y

    def create_basis_via_skinning_weights(self, rest_positions, assembly_ST, constraints, aux_size,
                                          use_pca=True, specify_verts=[]):
        """
        Complete Python equivalent of:
        PD::RHSInterpolationGroup::createBasisViaSkinningWeights
        """
        # (1) Compute rest-state auxiliaries (simplified, assuming S = identity if constraints already map properly)
        rest_state_aux = assembly_ST.T @ rest_positions.copy()

        check_matrix_health(rest_state_aux, "aux_rest")
        # (2) Build constraint-space skinning matrix Y

        Y = self.create_skinning_space_constraints(rest_state_aux, self.weights, constraints, aux_size)
        # Y = create_skinning_space(rest_state_aux, self.weights, constraints, aux_size, self.faces, self.tets)

        # n = rest_state_aux.shape[0]
        # # === Step 3: Build U matrix ===
        # row_idx = []
        # col_idx = []
        # data = []
        #
        # for i in range(n//aux_size):  # row
        #     for pi in range(aux_size):  # for each vertex in the element
        #         row = i * aux_size + pi
        #         for j in range(self.num_components): # col
        #             w = self.weights[i//aux_size, j]
        #             if w < 1e-6:
        #                 continue
        #
        #             row_idx.append(row)
        #             col_idx.append(j * 4 + 3)
        #             data.append(w)
        #
        #             # === Extra affine/shear component ===
        #             for d in range(3):
        #                 row_idx.append(row)
        #                 col_idx.append(j * 4 + d)
        #                 data.append(w * rest_state_aux[row,d])  # simplified dot with [1,1,1]
        #
        # Y = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, 4 * self.num_components)).tocsr()
        #
        # (4) Perform PCA or not
        if use_pca:
            self.V = self.snapshot_pca(Y, specify_verts)
        else:
            self.V = Y

        check_matrix_health(self.V, "constraint projection basis")

    def init_constraint_group_interpolation(self,
                           constraints,
                           assembly_matrix,
                           regularization_weight=0.0,
                           aux_size=1):
        """
        Python equivalent of PD::RHSInterpolationGroup::initInterpolation
        Builds interpolation system for reduced constraint projections.
        """

        n_constraints = len(constraints)
        # --------------------------------------------------------
        # Create selection and weight matrices
        # --------------------------------------------------------
        # Weight vector
        weights = np.ones(n_constraints * aux_size)
        for i, c in enumerate(constraints):
            for j in range(aux_size):
                idx = i * aux_size + j
                weights[idx] = c.wi   # constraint weight
        # W = diags(weights)

        # Selection matrix (sampled constraints only)
        rows, cols, vals = [], [], []
        sampled_constraints = []
        alpha = []

        # get ids of elements that touch any of the sampled verts
        self.get_sampled_constrained_elements()

        included_samples = []
        for i, c in enumerate(constraints):
            if c.id in self.sampled_constraints_ids and len(sampled_constraints) < len(self.vert_samples) :
                sampled_constraints.append(c)
                alpha.append(c.id)
                r = len(sampled_constraints) - 1
                for d in range(aux_size):
                    rows.append(r*aux_size+d)
                    cols.append(c.id*aux_size+d)
                    vals.append(1.0)

        J_sel = csr_matrix((vals, (rows, cols)),
                           shape=(aux_size*len(self.sampled_constraints_ids), aux_size*n_constraints))
        self.sampled_constraints = sampled_constraints
        # --------------------------------------------------------
        # Compute linear interpolation matrices
        # --------------------------------------------------------
        # S = assembly_matrix.tocsr()
        JTJ = J_sel.T @ J_sel


        if regularization_weight > 0 and self.mass is not None:
            lhs = self.V.T @ ((JTJ + regularization_weight * self.mass) @ self.V)
        else:
            lhs = self.V.T @ (JTJ @ self.V)

        rhs = self.V.T @ J_sel.T

        # Factorize LHS
        chol_L = cholesky(lhs + 1e-12 * np.eye(lhs.shape[0]), lower=True)

        # Store solver data
        self.interpol_solver = [chol_L, rhs]
        self.alpha_points = alpha
