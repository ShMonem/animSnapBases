import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags, vstack
from scipy.linalg import cholesky, solve, eigh, svd, lu_factor
# try:
#     import gdist
#     USE_GDIST = True
# except ImportError:
#     USE_GDIST = False
#     from pygeodesic import geodesic

import torch
from stvd import STVD, STVD_MODE_ST_DIJKSTRA, STVD_MODE_GRAPH_DIJKSTRA
import math
# from utils import compute_surface_geodesics
import igl
import polyscope as ps

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
            # for v in range(self.n_vertices):
            #     best_vert = np.argmax(self.distances)

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
        num_small_samples=-1, small_sample_radius=-1
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

        U = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n * 3, 4 * self.num_components)).tocsr()
        self.U = np.hstack((U.toarray(), np.ones((U.shape[0], 1))))

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
        self.mass = None   # geometrical masses (area/volume based)
        self.particles_mass = None   # dynamical masses (for particles motion)

        self.sampled_constraints_ids = []
        self.sampled_constraints= []
        self.interpol_solver = {}
        self.alpha_points = []
        self.r_multiplier =r_multiplier
        self.basis_scale = basis_scale

    def get_sampled_constrained_elements(self):

        self.sampler.extend_samples(self.num_samples, self.vert_samples)

        # # ps.init()
        # ps.remove_all_structures()

        # Register surface mesh
        # ps_mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)
        #
        # # Highlighted points as a point cloud
        # highlight_positions = self.vertices[self.vert_samples]
        # ps.register_point_cloud("highlighted verts", highlight_positions, radius=0.01)

        # ps.show()


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

            else:
                # fallback: assign all weight to max value group
                max_idx = np.argmax(tet_weights[t, :])
                tet_weights[t, :] = 0
                tet_weights[t, max_idx] = 1.0

        return tet_weights

    def compute_skinning_weights(self):
        max_distance = self.sampler.get_sample_diameter(self.vert_samples)
        weights = self.sampler.get_radial_base_functions(self.vert_samples, r=max_distance * self.r_multiplier,
                                                         partition_of_one=True)

        # ## ------------------------------
        # ## Step 2. Farthest-point sampling on the surface
        # ## ------------------------------
        # def farthest_point_sampling(V, F, k):
        #     """Return k approximately evenly distributed vertex indices using graph-geodesic FPS."""
        #     n = V.shape[0]
        #
        #     # Build adjacency graph from triangles
        #     adj = [[] for _ in range(n)]
        #     for tri in F:
        #         for i in range(3):
        #             a, b = tri[i], tri[(i + 1) % 3]
        #             adj[a].append(b)
        #             adj[b].append(a)
        #
        #     # Precompute edge lengths
        #     edge_lengths = {}
        #     for i in range(n):
        #         for j in adj[i]:
        #             edge_lengths[(i, j)] = np.linalg.norm(V[i] - V[j])
        #
        #     def dijkstra(src):
        #         """Compute shortest path (graph geodesic) distance from src."""
        #         import heapq
        #         dist = np.full(n, np.inf)
        #         dist[src] = 0.0
        #         pq = [(0.0, src)]
        #         while pq:
        #             d, u = heapq.heappop(pq)
        #             if d > dist[u]:
        #                 continue
        #             for v in adj[u]:
        #                 alt = d + edge_lengths[(u, v)]
        #                 if alt < dist[v]:
        #                     dist[v] = alt
        #                     heapq.heappush(pq, (alt, v))
        #         return dist
        #
        #     # Farthest point sampling
        #     seeds = [np.random.randint(0, n)]
        #     D = dijkstra(seeds[0])
        #     for _ in range(1, k):
        #         next_seed = np.argmax(D)
        #         seeds.append(next_seed)
        #         D = np.minimum(D, dijkstra(next_seed))
        #
        #     return np.array(seeds)
        # #
        # # seeds = farthest_point_sampling(self.vertices, self.faces, len(self.vert_samples))
        # # print("Selected seeds:", seeds)
        #
        # # ------------------------------
        # # Step 3. Compute Bounded Biharmonic Weights (BBW)
        # # ------------------------------
        # # Harmonic (smooth) weights fallback if BBWData is unavailable
        # seeds = np.array(self.vert_samples)
        # b = seeds
        # bc = np.eye(len(b))
        #
        # print("Computing harmonic weights (BBW fallback)...")
        # weights = igl.harmonic(self.vertices, self.faces, b, bc, 1)  # 2 = biharmonic, 1 = Laplacian
        # # weights = np.maximum(weights, 0)  # Clamp to positive
        # weights /= weights.sum(axis=1, keepdims=True)


        # ps_mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)
        #
        # # Register weights as scalar fields
        # for i in range(weights.shape[1]):
        #     ps_mesh.add_scalar_quantity(f"weight_{i}", weights[:, i], defined_on='vertices', enabled=(i == 0))
        #
        # # Register sampled handles
        # ps.register_point_cloud("handles", self.vertices[self.vert_samples], radius=0.01, color=(1, 0, 0))

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

    def compute_constraint_mass_matrix(self, group_name, constraints, vertex_masses, particle_masses, aux_size):
        """
        Compute mass matrix and its diagonal for the constraint projection auxiliaries.

        Args:
            particle_masses:
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
        M_diag_d = np.zeros(num_constraints * aux_size)


        if group_name == "verts_bending":
            self.mass = vertex_masses
            self.particles_mass = particle_masses
        else:
            for i, c in enumerate(constraints):
                w = np.sum(vertex_masses[c.indices])
                M_diag[i * aux_size:(i + 1) * aux_size] = w

                w_d = np.sum(particle_masses[c.indices])
                M_diag_d[i * aux_size:(i + 1) * aux_size] = w_d


            self.mass = M_diag
            self.particles_mass = M_diag_d

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
        # constraint_ids = torch.tensor([c.id for c in constraints], device=device, dtype=torch.long)

        # Gather weights for all constraints: shape (num_constraints, num_basis)
        W = skinning_weights  #[constraint_ids, :]  # (Nc, B)

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
            cur_w_expanded = cur_w.repeat_interleave(rows_per_constraint) # (Nc*aux_size,)

            # Apply weights to rest_state_aux
            Y[:, g * (dim + 1): g * (dim + 1) + dim] = rest_state_aux * cur_w_expanded.unsqueeze(1)
            Y[:, g * (dim + 1) + dim] = cur_w_expanded

        return Y

    def create_basis_via_skinning_weights(self, rest_positions, assembly_ST_no_weights, constraints, aux_size,
                                          use_pca=True, specify_verts=[], normalization_factor=1.0):
        """
        Complete Python equivalent of:
        PD::RHSInterpolationGroup::createBasisViaSkinningWeights
        """
        # Compute rest-state auxiliaries (simplified, assuming S = identity if constraints already map properly)
        rest_state_aux = torch.sparse.mm(assembly_ST_no_weights.transpose(0,1), torch.from_numpy(rest_positions).to(dtype=torch.float32, device=assembly_ST_no_weights.device))
        # Build constraint-space skinning matrix Y
        Y = self.create_skinning_space_constraints_torch(rest_state_aux, self.weights, constraints, aux_size)
        Y = torch.hstack((Y, torch.ones((Y.shape[0], 1), device=Y.device, dtype=Y.dtype)))

        # Perform PCA or not
        def mass_weighted_pca(Y, M_geom, mean_axis=0, tol_rel=1e-6, max_cols=-1):
            """
            Y: (m, s) snapshots (float, on same device)
            M: (m,) mass vector (float, same device)
            Returns V (m, r) with V^T M V = I
            """
            # 1) center snapshots (column-wise mean)
            Yc = Y - Y.mean(dim=mean_axis, keepdim=True)

            # 2) form A = Yc^T M Yc  (s x s)
            Mw = M_geom.unsqueeze(1)  # (m,1)
            A = Yc.T @ (Mw * Yc)  # symmetric PSD

            # 3) eigendecomp
            vals, vecs = torch.linalg.eigh(A)  # ascending
            # keep significant spectrum
            vmax = vals.max()
            keep = vals > (tol_rel * vmax)
            vals = vals[keep]
            vecs = vecs[:, keep]  # (s, r)

            # 4) spatial modes: V0 = Yc @ (vecs / sqrt(vals))
            V0 = Yc @ (vecs / torch.sqrt(vals).unsqueeze(0))  # (m, r)
            # 5) numerically re-orthonormalize w.r.t. M  (QR in geometrical mass metric)
            # Let L = sqrt(M). We do Euclidean QR on L^T V0 then back-weighting with L^T^{-1}.
            L = torch.sqrt(M)  # (m,)
            Qe, _ = torch.linalg.qr(V0 * L[:, None])  # Euclidean QR
            V = Qe / L[:, None]  # now V^T M V = I
            V = torch.hstack((V, torch.ones((Y.shape[0], 1), device=Y.device, dtype=Y.dtype)))

            return V #, vals[:max_cols]

        def metric_convert_basis(Vg, Mdyn, tol=1e-10):
            # Vg: (m,r), Mdyn: (m,)
            MV = Mdyn[:, None] * Vg  # (m,r)
            G = Vg.T @ MV  # (r,r), SPD (up to tiny modes)

            # drop tiny modes if needed
            evals, evecs = torch.linalg.eigh(G)
            keep = evals > tol * evals.max()
            Vg = Vg @ evecs[:, keep]
            Gk = torch.diag(evals[keep])
            r = Gk.shape[0]

            # Cholesky and RIGHT-multiply by R^{-1}
            R = torch.linalg.cholesky(Gk)  # (r,r), upper by default

            # Option A: triangular solve for right-multiply
            # V = Vg @ R^{-1}  <=>  solve(R^T, Vg^T)^T
            V = torch.linalg.solve(R.T, Vg.T).T

            # Option B (also fine for small r): explicit inv(R)
            # Rinvt = torch.linalg.inv(R)
            # V     = Vg @ Rinvt

            # sanity: V^T Mdyn V ≈ I
            # err = torch.linalg.norm((Mdyn[:,None]*V).T @ V - torch.eye(r, device=V.device))
            return V

        if use_pca:
            M = torch.as_tensor(self.mass, dtype=Y.dtype, device=Y.device)
            M_dyn = torch.as_tensor(self.particles_mass, dtype=Y.dtype, device=Y.device)

            # PCA basis
            self.V = mass_weighted_pca(Y, M_dyn, tol_rel=1e-8, max_cols=self.num_components) *self.basis_scale/(normalization_factor) #

        else:
            self.V = Y / (10*self.particles_mass.max())
        rank = torch.linalg.matrix_rank(self.V, tol=1e-6)
        print(f"Skinning basis_, shape {self.V.shape} and rank: {rank}")


    def init_constraint_group_interpolation(self,
                           constraints,
                           aux_size=1):
        """
        Python equivalent of PD::RHSInterpolationGroup::initInterpolation
        Builds interpolation system for reduced constraint projections.
        """

        n_constraints = len(constraints)
        # --------------------------------------------------------
        # Create selection and weight matrices
        # --------------------------------------------------------
        # # Weight vector
        # weights = np.ones(n_constraints * aux_size)
        # for i, c in enumerate(constraints):
        #     for j in range(aux_size):
        #         idx = i * aux_size + j
        #         weights[idx] = c.wi * mass_normalizarion  # constraint weight
        #
        # Wei = torch.from_numpy(weights).to(device=self.V.device, dtype=self.V.dtype)

        # Selection matrix (sampled constraints only)
        rows, cols, vals = [], [], []
        # sampled_constraints = []

        # get ids of elements that touch any of the sampled verts
        self.get_sampled_constrained_elements()

        def build_selection_matrix_torch(constraints, sampled_constraints_ids, aux_size, n_constraints, device="cuda"):
            # Keep the exact *constraints iteration order* (like your original loop)
            sampled_set = set(sampled_constraints_ids)
            ids_in_constraints_order = sampled_constraints_ids #[c.id for c in constraints if c.id in sampled_constraints_ids]
            self.sampled_constraints = [constraints[idx] for idx in sampled_constraints_ids]
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
        J_sel = J_sel.to(self.V.device)
        rank = torch.linalg.matrix_rank(J_sel.to_dense(), tol=1e-6)
        print(f"J_sel, shape {J_sel.shape} and  rank: {rank}")
        # --------------------------------------------------------
        # Compute interpolation matrices
        # --------------------------------------------------------
        # Compute JTJ
        JTJ = torch.sparse.mm(J_sel.T, J_sel) # result is dense tensor

        # Compute lhs = V^T (JTJ V)
        lhs = self.V.T @ (JTJ @ self.V)

        rank = torch.linalg.matrix_rank(lhs, tol=1e-6)
        print(f"V^T J^T J V, , shape {lhs.shape} and  rank: {rank}")

        # Compute rhs = V^T J_sel^T
        rhs = torch.sparse.mm(self.V.T, J_sel.T)

        rank = torch.linalg.matrix_rank(lhs, tol=1e-6)
        print(f"V^T J^T, , shape {rhs.shape} and  rank: {rank}")

        eps = 1e-8
        max_eps = 1e-2
        def chol(A, eps):
            while True:
                try:
                    lam = eps *  A.diag().mean()
                    L = torch.linalg.cholesky(A + lam * torch.eye(A.shape[0], device=lhs.device, dtype=A.dtype))
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
        # lhs_np = lhs.to_dense().cpu().detach().contiguous().numpy()
        # solver = lu_factor(lhs_np + eps * np.eye(lhs_np.shape[0]))
        # self.interpol_solver = [solver, rhs.to_dense().cpu().detach().numpy()]

        # self.V = self.V * Wei.unsqueeze(1)
