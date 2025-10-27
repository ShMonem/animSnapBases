import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, vstack
from scipy.linalg import cholesky, solve, eigh, svd, lu_factor
try:
    import gdist
    USE_GDIST = True
except ImportError:
    USE_GDIST = False
    from pygeodesic import geodesic

import torch
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

    if np.isnan(weights).any() or np.isnan(rest_positions).any():
        raise ValueError(" Warning: NaN detected in weights or rest positions.")

    k = weights.shape[1]
    num_rows = rest_positions.shape[0]
    dim = rest_positions.shape[1] - (1 if flat_space else 0)
    skinning_space = np.zeros((num_rows, (dim + 1) * k ), dtype=float)

    error = False

    for v in range(num_rows):
        for g in range(k):

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
    # skinning_space[:, -1] = 1.0

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
        self.stvd.is_updated = False
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
            max_dist = 0
            for v in range(self.n_vertices):
                best_vert = np.argmax(self.distances)

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
    def __init__(self, r_multiplier, vertices, faces=None, tets=None, num_samples=-1):
        self.U = None
        self.rest_pos = vertices
        self.faces = faces
        self.tets = tets
        self.sampler = Sampler(vertices, faces, tets)
        self.vert_samples = self.sampler.get_samples(num_samples)
        self.num_components = num_samples
        self.weights = None
        self.r_multiplier =r_multiplier

    def compute_skinning_weights(self):
        # Functions to compute LBS-subspace for positions
        """
        Compute subspace matrix U ∈ R^{n×4k} for position encoding.

        Parameters:
            vertices: (n, 3) array of vertex positions
            k: number of sample points (i.e. handles)
            r_factor: scaling for support radius r

        Returns:
            U: (n, 4k) sparse matrix
        """
        # === Step 1: Furthest Point Sampling ===

        max_distance = self.sampler.get_sample_diameter(self.vert_samples)
        weights = self.sampler.get_radial_base_functions(self.vert_samples, r = max_distance * self.r_multiplier, partition_of_one=True)

        # Normalize weights so that they sum to 1 per vertex
        weight_sums = np.sum(weights, axis=1, keepdims=True) + 1e-8
        weights /=weight_sums

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
    def __init__(self, r_multiplier, basis_scale, constraint_name, vertices, faces=None, tets=None, num_components=-1, num_samples=-1):
        self.V = None
        self.vertices = vertices
        self.faces = faces
        self.tets = tets
        self.sampler = Sampler(vertices, faces, tets)
        self.num_components = num_components
        self.vert_samples = self.sampler.get_samples(self.num_components)
        self.num_samples = num_samples
        self.constraint_name = constraint_name
        self.weights = None
        self.mass = None
        self.sampled_constraints_ids = []
        self.interpol_solver = {}
        self.alpha_points = []
        self.r_multiplier =r_multiplier
        self.basis_scale = basis_scale

    def get_sampled_constrained_elements(self):

        self.sampler.extend_samples(self.num_samples, self.vert_samples)
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
            if row_sum > 1e-10:
                tet_weights[t, :] /= row_sum
        return tet_weights

    def compute_skinning_weights(self):
        max_distance = self.sampler.get_sample_diameter(self.vert_samples)
        weights = self.sampler.get_radial_base_functions(self.vert_samples, r=max_distance * self.r_multiplier,
                                                         partition_of_one=True)
        #
        # # Normalize weights so that they sum to 1 per vertex
        weight_sums = np.sum(weights, axis=1, keepdims=True) + 1e-8
        weights /= weight_sums

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
        Perform mass-weighted PCA
        Args:
            Y: (m, s) snapshot matrix (each column = constraint-space feature)
            masses_diag: (m, ) mass weights vector
            size: number of principal components to keep

        Returns:
            base: (m, size+1) reduced PCA basis
        """
        # Step 1: remove column-wise mean
        Y_centered = Y - Y.mean(axis=1, keepdims=True)

        def robust_std(x):
            if isinstance(x, torch.Tensor):
                return x.std()
            else:
                return np.std(x)

        factor = 1.0 / robust_std(Y_centered)

        Y_centered *=factor

        mass = torch.from_numpy(self.mass).to(device=Y_centered.device, dtype=Y_centered.dtype)
        A = Y_centered.T @ (mass.unsqueeze(1) * Y_centered)
        # Step 3: eigendecomposition
        # Step 1: Eigen-decomposition (symmetric matrix A)
        vals, vecs = torch.linalg.eigh(A)  # vals: (n,), vecs: (n,n)

        # Step 2: Normalize eigenvalues by their max
        temp = vals / vals.max()

        # # Step 3: Keep only significant eigenvalues (> tol)
        mask = temp > 1e-14
        vals = vals[mask]
        vecs = vecs[:, mask]

        # Step 4: Sort in descending order
        sorted_vals, idx = torch.sort(vals, descending=True)
        vecs = vecs[:, idx]
        vals = sorted_vals

        # Step 5: Compute PCA-style bases
        base = Y_centered @ (vecs / torch.sqrt(vals.unsqueeze(0)))  # broadcast sqrt(vals)

        V = torch.hstack((base, torch.ones((base.shape[0], 1), device=Y.device, dtype=Y.dtype)))
        V /= factor
        V += Y.mean(axis=1, keepdims=True)
        return V

    def create_skinning_space_constraints(self, rest_state_aux, skinning_weights, constraints, aux_size):

        """
        Equivalent of PD::createSkinningSpace with forConstraints != nullptr
        Builds the constraint-space skinning matrix Y.
        """
        num_basis = skinning_weights.shape[1]
        num_rows, dim = rest_state_aux.shape
        Y = np.zeros((num_rows, num_basis * (dim + 1)))

        for i, c in enumerate(constraints):
            for g in range(num_basis):
                # Find which element the constraint refers to
                cur_weight = skinning_weights[c.id, g]
                if cur_weight < 1e-8:
                    continue
                for v in range(aux_size):
                    row = i * aux_size + v
                    for d in range(dim):
                        Y[row, g * (dim + 1) + d] = rest_state_aux[row, d] * cur_weight
                    Y[row, g * (dim + 1) + dim] = cur_weight

        return Y

    def create_skinning_space_constraints_torch(self, rest_state_aux, skinning_weights, constraints, aux_size):
        """
        Vectorized PyTorch version of create_skinning_space_constraints.
        Builds the constraint-space skinning matrix Y (|constraints|*aux_size, num_basis*(dim+1)).
        """

        device = rest_state_aux.device if isinstance(rest_state_aux, torch.Tensor) else 'cpu'
        rest_state_aux = torch.as_tensor(rest_state_aux, dtype=torch.float32, device=device)
        skinning_weights = torch.as_tensor(skinning_weights, dtype=torch.float32, device=device)

        num_basis = skinning_weights.shape[1]
        num_rows, dim = rest_state_aux.shape
        num_constraints = len(constraints)

        # Preallocate Y on GPU
        Y = torch.zeros((num_rows, num_basis * (dim + 1)), device=device, dtype=torch.float32)

        # Extract constraint IDs once
        constraint_ids = torch.tensor([c.id for c in constraints], device=device, dtype=torch.long)

        # Gather weights for all constraints: shape (num_constraints, num_basis)
        W = skinning_weights[constraint_ids, :]  # (Nc, B)

        # Reshape rest_state_aux to match constraint structure
        # rest_state_aux is already (num_rows, dim), rows = Nc * aux_size
        # Compute the row indices for each constraint’s aux block
        rows_per_constraint = aux_size

        # Broadcast over basis functions
        for g in range(num_basis):
            # weights for this basis across constraints
            cur_w = W[:, g]  # (Nc,)
            mask = cur_w > 1e-8
            if not mask.any():
                continue

            # Repeat each constraint’s weight for aux_size rows
            cur_w_expanded = cur_w.repeat_interleave(rows_per_constraint)  # (Nc*aux_size,)

            # Apply weights to rest_state_aux
            Y[:, g * (dim + 1):g * (dim + 1) + dim] = rest_state_aux * cur_w_expanded.unsqueeze(1)
            Y[:, g * (dim + 1) + dim] = cur_w_expanded

        return Y

    def create_basis_via_skinning_weights(self, rest_positions, assembly_ST, constraints, aux_size,
                                          use_pca=True, specify_verts=[]):
        """
        Complete Python equivalent of:
        PD::RHSInterpolationGroup::createBasisViaSkinningWeights
        """
        # Compute rest-state auxiliaries (simplified, assuming S = identity if constraints already map properly)
        rest_state_aux = torch.sparse.mm(assembly_ST.transpose(0,1), torch.from_numpy(rest_positions).to(dtype=torch.float32, device=assembly_ST.device))
        # Build constraint-space skinning matrix Y
        Y = self.create_skinning_space_constraints_torch(rest_state_aux, self.weights, constraints, aux_size)

        # Perform PCA or not
        if use_pca:
            Y = self.snapshot_pca(Y, specify_verts)

        # Final column is constant
        self.V = torch.hstack((Y, torch.ones((Y.shape[0], 1), device=Y.device, dtype=Y.dtype)))
        factor = 1/(5*self.V.norm())
        # print(1/(10*self.V.norm()))
        self.V *= factor   # normalize to avoid blowup

    def init_constraint_group_interpolation(self,
                           constraints,
                           aux_size=1,
                            mass_normalizarion=1.0):
        """
        Python equivalent of PD::RHSInterpolationGroup::initInterpolation
        Builds interpolation system for reduced constraint projections.
        """

        n_constraints = len(constraints)
        # --------------------------------------------------------
        # Create selection and weight matrices
        # --------------------------------------------------------
        # # Weight vector
        weights = np.ones(n_constraints * aux_size)
        for i, c in enumerate(constraints):
            for j in range(aux_size):
                idx = i * aux_size + j
                weights[idx] = c.wi * mass_normalizarion  # constraint weight

        Wei = torch.from_numpy(weights).to(device=self.V.device, dtype=self.V.dtype)

        # Selection matrix (sampled constraints only)
        rows, cols, vals = [], [], []
        sampled_constraints = []

        # get ids of elements that touch any of the sampled verts
        self.get_sampled_constrained_elements()

        def build_selection_matrix_torch(constraints, sampled_constraints_ids, aux_size, n_constraints, device="cuda"):
            # Keep the exact *constraints iteration order* (like your original loop)
            sampled_set = set(sampled_constraints_ids)
            ids_in_constraints_order = [c.id for c in constraints if c.id in sampled_set]
            n_sampled = len(ids_in_constraints_order)

            if n_sampled == 0:
                # empty sparse COO
                size = (0, aux_size * n_constraints)
                return torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=torch.float32, device=device),
                    size=size, device=device
                ).coalesce()

            ids_t = torch.tensor(ids_in_constraints_order, dtype=torch.long, device=device)

            # Row indices: r*aux_size + d, with r in [0..n_sampled-1], d in [0..aux_size-1]
            r = torch.arange(n_sampled, device=device).repeat_interleave(aux_size)
            d = torch.arange(aux_size, device=device).repeat(n_sampled)
            rows = r * aux_size + d  # shape: (n_sampled*aux_size,)

            # Col indices: c.id*aux_size + d, using the encountered ids order
            cols = (ids_t.repeat_interleave(aux_size) * aux_size) + d  # same shape

            vals = torch.ones_like(rows, dtype=torch.float32)

            J_sel = torch.sparse_coo_tensor(
                torch.stack([rows, cols], dim=0),  # (2, nnz)
                vals,
                size=(aux_size * n_sampled, aux_size * n_constraints),
                device=device,
                dtype=torch.float32
            ).coalesce()

            return J_sel

        J_sel = build_selection_matrix_torch(constraints, self.sampled_constraints_ids, aux_size, n_constraints )
        self.sampled_constraints = sampled_constraints

        # --------------------------------------------------------
        # Compute interpolation matrices
        # --------------------------------------------------------
        # Compute JTJ
        JTJ = torch.sparse.mm(J_sel.T, J_sel)  # result is dense tensor

        # Compute lhs = V^T (JTJ V)
        lhs = self.V.T @ (JTJ @ self.V)

        # Compute rhs = V^T J_sel^T
        rhs = torch.sparse.mm(self.V.T, J_sel.T)

        eps = 1e-8
        max_eps = 1e-2
        def chol(A, eps):
            while True:
                try:
                    L = torch.linalg.cholesky(A + eps * torch.eye(A.shape[0], device=lhs.device, dtype=A.dtype))
                    return L
                except RuntimeError as e:
                    if "not positive definite" not in str(e):
                        raise
                    if eps > max_eps:
                        raise RuntimeError(f"Matrix not PD even after adding eps={eps}")
                    eps *= 0.5  # increase regularization and retry

        # Factorize LHS
        chol_L= chol(lhs, eps)

        # Store solver data
        self.interpol_solver = [chol_L.to_dense().cpu().detach().numpy(), rhs.to_dense().cpu().detach().numpy()]

        self.V = self.V * Wei.unsqueeze(1)
