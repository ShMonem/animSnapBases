# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import trimesh
import numpy as np
from scipy.sparse import coo_matrix
from abc import ABC, abstractmethod
from igl import edges, boundary_facets, barycenter, winding_number, copyleft
from igl.copyleft import tetgen
from numpy.linalg import svd, det, inv
from scipy.sparse import lil_matrix, save_npz
from torch.ao.nn.quantized.functional import threshold

from utils import delete_matching_column
from dataclasses import dataclass
from joblib import Parallel, delayed

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

# Helper container for an edge and its triangles
@dataclass
class Edge:
    v2: int  # Neighbor vertex
    vOtherT1: int  # Third vertex in triangle 1
    t1: int  # Triangle 1 index
    vOtherT2: int = -1  # Third vertex in triangle 2 (optional)
    t2: int = -1  # Triangle 2 index (optional)


class Constraint(ABC):
    def __init__(self, id, indices, wi=1.0):
        self._indices = list(indices)  # list of vertex indices (ints)
        self._wi = wi                  # weight (float)
        self._selection_matrix = None   # differential operator and selection matrix to map from constriants projections to positions
        self._selection_matrix_no_weights = None
        self._pi = None  # projection to constraint manifold
        self._id = id

    @property
    def indices(self):
        return self._indices

    @property
    def wi(self):
        return self._wi

    @property
    def id(self):
        return self._id

    @property
    def selection_matrix(self):
        return self._selection_matrix

    @property
    def selection_matrix_no_weights(self):
        return self._selection_matrix_no_weights

    def evaluate(self, positions, masses):
        # Default is zero energy; override in subclass if needed
        return 0.0

    @abstractmethod
    def build_SiT(self, position_dim):
        "build the differential operator"
        pass

    @abstractmethod
    def get_pi(self, position, frame):
        "compute constraint projection for element i"
        pass

    @abstractmethod
    def project_wi_SiT_pi(self, q, rhs):
        """compute element constraint projection and map back to positions dimension"""
        pass
    # @abstractmethod
    # def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
    #     """
    #     Compute projection and update rhs.
    #     q: full position vector (flattened)
    #     rhs: numpy array to accumulate contributions
    #     """
    #     pass

    @abstractmethod
    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Return list of triplets (i, j, value) to be added to system matrix.
        """
        pass

class PositionalConstraint(Constraint):
    def __init__(self, id, indices, wi, positions, motion_type='fixed', frames_series=None, frame_reset=0):
        super().__init__(id, indices, wi)
        self.name = "positional"
        assert len(indices) == 1
        vi = indices[0]
        self.p0 = positions[vi].reshape(3, 1)  # Column vector  this can be current positions or init positions as given
        # build differential operator SiT
        self.build_SiT(positions.shape[0])
        self.type = motion_type
        
        if self.type == "user_defined" and frames_series is not None:
            self.position_shift = frames_series
            self.frame_reset = frame_reset

    def build_SiT(self, position_dim):
        self._selection_matrix = lil_matrix((position_dim, 1))
        self._selection_matrix_no_weights = self._selection_matrix.copy()
        self._selection_matrix[self.indices[0]] = self.wi

    def get_pi(self, q, frame=0):
        if self.type == 'fixed':
            return self.p0.reshape(-1,3)
        elif self.type == "user_defined":
            return self.p0.reshape(-1, 3) + self.position_shift[frame- self.frame_reset]
        else:
            raise ValueError(f"Unknown positional constaint {self.type}!")

    def project_wi_SiT_pi(self, q, rhs):
        rhs += self.selection_matrix @ self.get_pi(q)


    def get_wi_SiT_AiT_Ai_Si(self):
        vi = self.indices[0]
        triplets = [
            (3 * vi + 0, 3 * vi + 0, self.wi),
            (3 * vi + 1, 3 * vi + 1, self.wi),
            (3 * vi + 2, 3 * vi + 2, self.wi),
        ]
        return triplets

class VertBendingConstraint(Constraint):
    def __init__(self, id, v_ind, wi, v_star, voronoi_area, positions, triangles,
                 prevent_bending_flips=True, flat_bending=False):
        super().__init__(id, [v_ind], wi * voronoi_area)
        self.name = "verts_bending"
        self.v_ind = v_ind
        self.prevent_bending_flips = prevent_bending_flips
        self.flat_bending = flat_bending
        self.init_positions = positions
        self.init_triangles = triangles
        self.vertex_star = v_star
        self.voronoi_area = voronoi_area
        

        # # build differential operator SiT
        self.set_wi(wi* voronoi_area)  # TODO: test!
        self.build_SiT(positions.shape[0])

    def set_wi(self, value):
        self._wi = value

    def build_SiT(self, position_dim):
        # === Compute cotangent weights and local triangle list
        cotan_weights = []
        triangles_seen = set()
        self.triangles = []

        def compute_angle(a, b, c):
            """Compute angle at vertex b between edges (a-b) and (c-b)."""
            u = a - b
            v = c - b
            dot = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)
            return np.arccos(dot)

        def get_triangle_normal(tris, pos):
            """Return average normal of tris (for stability)."""
            normals = []
            for tri in tris:
                a, b, c = pos[tri]
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-10:
                    normals.append(n / n_norm)
            return np.mean(normals, axis=0) if normals else np.array([0, 0, 1])

        for edge in self.vertex_star:
            p0 = self.init_positions[self.v_ind]
            p1 = self.init_positions[edge.vOtherT1]
            p2 = self.init_positions[edge.v2]

            angle1 = compute_angle(p0, p1, p2)
            cot = 0.5 / np.tan(angle1)

            if edge.t2 >= 0:
                p1_2 = self.init_positions[edge.vOtherT2]
                angle2 = compute_angle(p0, p1_2, p2)
                cot += 0.5 / np.tan(angle2)

            cotan_weights.append(cot / self.voronoi_area)

            for t in [edge.t1, edge.t2]:
                if t >= 0 and t not in triangles_seen:
                    self.triangles.append(self.init_triangles[t])
                    triangles_seen.add(t)

        self.cotan_weights = np.array(cotan_weights)

        # === Compute rest mean curvature vector
        mean_curvature = np.zeros(3)
        for edge, w in zip(self.vertex_star, self.cotan_weights):
            mean_curvature += (self.init_positions[self.v_ind] - self.init_positions[edge.v2]) * w

        self.rest_mean_curvature = 0.0 if self.flat_bending else np.linalg.norm(mean_curvature)

        self.tri_normal = get_triangle_normal(np.array(self.triangles), self.init_positions)
        self.dot_with_normal = self.tri_normal @ mean_curvature

        # === Sparse selection matrix as triplet format
        nV = self.init_positions.shape[0]

        # --- Build row, col, val triplets directly
        row_idx = [self.v_ind] + [edge.v2 for edge in self.vertex_star]
        col_idx = [0] * len(row_idx)
        vals = [np.sum(self.cotan_weights)] + [-w for w in self.cotan_weights]

        # --- Create sparse column vector (COO)
        self._selection_matrix_no_weights = coo_matrix(
            (vals, (row_idx, col_idx)),
            shape=(nV, 1)
        )

        # Weighted version
        self._selection_matrix = coo_matrix(
            (self.wi * np.array(vals), (row_idx, col_idx)),
            shape=(nV, 1)
        )

    def get_pi(self, q, frame=0):

        v = self.v_ind
        star_sum = np.zeros(3)
        for edge, w in zip(self.vertex_star, self.cotan_weights):
            star_sum += (q[3 * v:3 * v + 3] - q[3 * edge.v2:3 * edge.v2 + 3]) * w

        norm = np.linalg.norm(star_sum)
        if norm < 1e-10:
            correction = self.tri_normal * self.rest_mean_curvature
        else:
            correction = star_sum * (self.rest_mean_curvature / norm)

        if self.prevent_bending_flips:
            dot = self.tri_normal @ correction
            if norm > 1e-5 and dot * self.dot_with_normal < 0:
                correction *= -1

        return correction.reshape(-1,3)

    def project_wi_SiT_pi(self, q, rhs):
        # if self.rest_mean_curvature < 1e-12:
        #     return

        rhs += self.selection_matrix @ self.get_pi(q)

    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Builds the contribution of the bending constraint to the global LHS matrix.
        Equivalent to: weight * Sᵢᵗ Aᵢᵗ Aᵢ Sᵢ = weight * Sᵢᵗ Sᵢ (since Ai = I)
        """
        # (1 x N) * (N x 1) = scalar
        S = self.selection_matrix  # shape: (1, num_vertices), sparse or dense
        ST = self.selection_matrix.T  # shape: (num_vertices, 1)

        # Matrix product: K = Sᵀ S (shape: num_vertices x num_vertices)
        K = ST @ S  # dense or sparse depending on format

        # Multiply by weight
        K = self.wi * K

        # Extract triplets: (i, j, val) for sparse matrix assembly
        triplets = []
        rows, cols = K.shape
        for i in range(rows):
            for j in range(cols):
                val = K[i, j]
                if abs(val) > 1e-12:
                    triplets.append((i * 3 + 0, j * 3 + 0, val))
                    triplets.append((i * 3 + 1, j * 3 + 1, val))
                    triplets.append((i * 3 + 2, j * 3 + 2, val))

        return triplets
    # def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
    #     """Applies the projection term to the global RHS."""
    #     if self.rest_mean_curvature < 1e-12:
    #         return
    #
    #     v = self.v_ind
    #     star_sum = np.zeros(3)
    #     for edge, w in zip(self.vertex_star, self.cotan_weights):
    #         star_sum += (q[3 * v:3 * v + 3] - q[3 * edge.v2:3 * edge.v2 + 3]) * w
    #
    #     norm = np.linalg.norm(star_sum)
    #     if norm < 1e-10:
    #         correction = self.tri_normal * self.rest_mean_curvature
    #     else:
    #         correction = star_sum * (self.rest_mean_curvature / norm)
    #
    #     if self.prevent_bending_flips:
    #         dot = self.tri_normal @ correction
    #         if norm > 1e-5 and dot * self.dot_with_normal < 0:
    #             correction *= -1
    #
    #     val = self.selection_matrix[self.v_ind]
    #     rhs[3 * self.v_ind: 3 * self.v_ind + 3] += self.wi * val * correction

class EdgeSpringConstraint(Constraint):
    def __init__(self, id, indices, wi, positions):
        super().__init__(id, indices, wi)
        self.name = "edge_spring"
        assert len(indices) == 2
        v0, v1 = indices[0], indices[1]
        self.d = np.linalg.norm(positions[v0] - positions[v1])

        # build differential operator SiT
        self.set_wi(wi * self.d)   # TODO: test
        self.build_SiT(positions.shape[0])

    def set_wi(self, value):
        self._wi = value

    def build_SiT(self, position_dim):
        """
        Build the edge selection matrix (S_i^T) and its no-weight version directly in COO format.

        position_dim : int
            total number of vertex DOFs (|V| or 3|V| depending on dimensionality)
        """
        # two nonzero entries: -1 at vertex i0, +1 at vertex i1
        row_idx = [self.indices[0], self.indices[1]]
        col_idx = [0, 0]  # single column
        vals = [-1.0, 1.0]

        # create sparse column vector in COO format
        self._selection_matrix_no_weights = coo_matrix(
            (vals, (row_idx, col_idx)),
            shape=(position_dim, 1)
        )

        # weighted version
        self._selection_matrix = coo_matrix(
            ([self.wi * v for v in vals], (row_idx, col_idx)),
            shape=(position_dim, 1)
        )


    def get_pi(self, q, frame=0):
        """
        Args:
            q: np.ndarray of shape (3*N,) – flattened positions
            rhs: np.ndarray of shape (N, 3) – accumulates constraint projection forces
        """
        vi, vj = self.indices
        p1 = q[3 * vi:3 * vi + 3]
        p2 = q[3 * vj:3 * vj + 3]
        spring = p2 - p1
        length = np.linalg.norm(spring)

        if length == 0:
            return  # Avoid divide by zero

        normalized_edge = spring / length
        delta = 0.5 * (length - self.d)
        pi1 = p1 + delta * normalized_edge
        pi2 = p2 - delta * normalized_edge

        self._pi = 0.5 * (pi2 - pi1).reshape(-1,3)
        return self._pi

    def project_wi_SiT_pi(self, q, rhs):
        """
        Args:
            q: np.ndarray of shape (3*N,) – flattened positions
            rhs: np.ndarray of shape (N, 3) – accumulates constraint projection forces
        """
        rhs += self.selection_matrix @ self.get_pi(q)

    def get_wi_SiT_AiT_Ai_Si(self):
        vi, vj = self.indices
        triplets = []
        w = self.wi * 0.5

        for i in range(3):
            triplets.append((3 * vi + i, 3 * vi + i, w))
            triplets.append((3 * vj + i, 3 * vj + i, w))
            triplets.append((3 * vi + i, 3 * vj + i, -w))
            triplets.append((3 * vj + i, 3 * vi + i, -w))

        return triplets

    # def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
    #     vi, vj = self.indices
    #     p1 = q[3 * vi:3 * vi + 3]
    #     p2 = q[3 * vj:3 * vj + 3]
    #     spring = p2 - p1
    #     length = np.linalg.norm(spring)
    #
    #     if length == 0:
    #         return  # Avoid divide by zero
    #
    #     normalized_edge = spring / length
    #     delta = 0.5 * (length - self.d)
    #     pi1 = p1 + delta * normalized_edge
    #     pi2 = p2 - delta * normalized_edge
    #
    #     rhs[3 * vi:3 * vi + 3] += self.wi * 0.5 * (pi1 - pi2)    # with -w
    #     rhs[3 * vj:3 * vj + 3] += self.wi * 0.5 * (pi2 - pi1)

class TriStrainConstraint(Constraint):
    def __init__(self, id, indices, wi, positions, sigma_min, sigma_max):
        super().__init__(id, indices, wi)
        assert len(indices) == 3
        self.name = "tris_strain"

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        

        v1, v2, v3 = indices
        p1 = positions[v1]
        p2 = positions[v2]
        p3 = positions[v3]

        # Define local 2D basis: P is 3x2 matrix
        e1 = p2 - p1
        e2 = p3 - p1

        P = np.zeros((3, 2))
        P[:, 0] = e1 / np.linalg.norm(e1)
        P[:, 1] = e2 - np.dot(e2, P[:, 0]) * P[:, 0]
        P[:, 1] /= np.linalg.norm(P[:, 1])

        self.P = P  # 3x2 projection to local tangent plane

        # Rest shape in local frame (2x2)
        rest_edges = np.column_stack([p2 - p1, p3 - p1])  # (3x2)
        rest_edges_2d = P.T @ rest_edges  # (2x2)
        self.DmInv = np.linalg.inv(rest_edges_2d)  # inverse of 2D rest matrix

        self.A0 = 0.5 * np.linalg.det(rest_edges_2d)  # reference area in local 2D frame

        # build differential operator SiT
        self.set_wi(wi * abs(self.A0))  # TODO: test
        self.build_SiT(positions.shape[0])

    def set_wi(self, value):
        self._wi = value

    def build_SiT(self, num_vertices):
        """
        Constructs and caches the transposed selection matrix (Si^T) for the triangle.
        Si^T shape: (num_vertices, 2), sparse
        """
        v1, v2, v3 = self.indices
        # Local shape matrix (2×3)
        G = np.column_stack([self.DmInv.T, -np.sum(self.DmInv.T, axis=1)])  # (2, 3)

        # Each triangle contributes 3 vertices × 2 columns = 6 nonzeros
        row_idx = np.array([v1, v2, v3, v1, v2, v3])  # vertex IDs repeated for 2 columns
        col_idx = np.array([0, 0, 0, 1, 1, 1])  # which column (x/y component)
        vals = np.hstack([G[0, :], G[1, :]])  # flatten G row-wise

        # Create sparse COO matrices
        self._selection_matrix_no_weights = coo_matrix(
            (vals, (row_idx, col_idx)),
            shape=(num_vertices, 2)
        )

        # Weighted version
        self._selection_matrix = coo_matrix(
            (self.wi * vals, (row_idx, col_idx)),
            shape=(num_vertices, 2)
        )


    def get_pi(self, q, frame=0):
        """
        Computes the projection target pi in 2D strain space (shape 2x3).
        """
        v1, v2, v3 = self.indices
        q1 = q[3 * v1:3 * v1 + 3]
        q2 = q[3 * v2:3 * v2 + 3]
        q3 = q[3 * v3:3 * v3 + 3]

        Ds = np.column_stack([q2 - q1, q3 - q1])  # shape (3, 2)
        Ds_2d = self.P.T @ Ds  # project to 2D: shape (2, 2)

        U, s, Vt = svd(Ds_2d @ self.DmInv)
        s = np.clip(s, self.sigma_min, self.sigma_max)
        Fhat = U @ np.diag(s) @ Vt  # shape (2, 2)

        # G = np.column_stack([self.DmInv.T, -np.sum(self.DmInv.T, axis=1)])  # shape (2, 3)

        pi = (self.P @ Fhat).T  # shape (2, 3)
        return pi

    def project_wi_SiT_pi(self, q, rhs):
        rhs += self.selection_matrix @ self.get_pi(q)

    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Builds the contribution of a triangle constraint to the global LHS matrix.
        """

        grads = self.DmInv.T  # shape: (2, 2)
        grads_l = -np.sum(grads, axis=0)  # shape: (2,)

        G = np.column_stack([grads.T, grads_l])  # shape: (2, 3)

        # G (2x3) --> G.T (3x2) x G (2x3) = (3x3)
        K3x3 = G.T @ G

        K9x9 = np.kron(K3x3, np.eye(3)) * (self.wi)

        triplets = []
        for i in range(9):
            for j in range(9):
                val = K9x9[i, j]
                if abs(val) > 1e-12:
                    row = 3 * self.indices[i // 3] + i % 3
                    col = 3 * self.indices[j // 3] + j % 3
                    triplets.append((row, col, val))

        return triplets

    # def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
    #     """
    #     Adds constraint projection term to the global RHS vector (in 3D).
    #     """
    #     v1, v2, v3 = self.indices
    #     q1 = q[3 * v1:3 * v1 + 3]
    #     q2 = q[3 * v2:3 * v2 + 3]
    #     q3 = q[3 * v3:3 * v3 + 3]
    #
    #     Ds = np.column_stack([q2 - q1, q3 - q1])  # 3x2
    #     Ds_2d = self.P.T @ Ds  # Projected deformation matrix (2x2)
    #
    #     U, s, Vt = svd(Ds_2d @ self.DmInv)
    #     s = np.clip(s, self.sigma_min, self.sigma_max)
    #     Fhat = U @ np.diag(s) @ Vt  # Projected strain (2x2)
    #
    #     G = np.column_stack([self.DmInv.T, -np.sum(self.DmInv.T, axis=1)])  # (2, 3)
    #
    #     correction_2d = Fhat @ G  # (2, 3)
    #     correction_3d = self.P @ correction_2d  # (3, 3)
    #
    #     weight = self.wi * abs(self.A0)
    #
    #     for i, idx in enumerate(self.indices):
    #         rhs[3 * idx:3 * idx + 3] += weight * correction_3d[:, i]

class TetStrainConstraint(Constraint):
    """Constrains only the stretch components of the deformation — that is, the singular values σ₁, σ₂, σ₃ of F
    Allows arbitrary rotation but limits stretching and compression (by clamping σ). """
    def __init__(self, id, indices, wi, positions, sigma_min, sigma_max):
        super().__init__(id, indices, wi)
        assert len(indices) == 4
        self.name = "tets_strain"

        # self.indices =  list(indices)
        # self.wi = wi
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        

        v1, v2, v3, v4 = indices
        p1 = positions[v1]
        p2 = positions[v2]
        p3 = positions[v3]
        p4 = positions[v4]

        Dm = np.column_stack([p1 - p4, p2 - p4, p3 - p4])  # 3x3 matrix, cols: [p1-p4, p2-p4, p3-p4]

        self.DmInv = np.linalg.inv(Dm)  # inverse of reference shape matrix (helps to detect inverted tets)
        self.V0 = (1.0 / 6.0) * np.linalg.det(Dm)  # undeformed tetrahedron volume

        # build differential operator SiT
        self.set_wi(wi * abs(self.V0))  # TODO: test
        self.build_SiT(positions.shape[0])

    def set_wi(self, value):
        self._wi = value

    def build_SiT(self, num_vertices):
        """
        Precomputes S_i^T: selection matrix transposed for this tet.
        SiT: shape (num_vertices, 4), each column corresponds to one tet vertex.
        We build a 3D gradient matrix of shape (num_vertices, 3), per-vertex.
        """
        grads = self.DmInv  # shape (3, 3)
        grads_l = -np.sum(grads, axis=0)  # shape (3,)

        G = np.column_stack([grads.T, grads_l])  # shape (3, 4)

        # Suppose: num_vertices = |V|, G is (3, 4)
        v1, v2, v3, v4 = self.indices
        rows = np.array([v1, v2, v3, v4] * 3)
        cols = np.repeat(np.arange(3), 4)  # [0,0,0,0, 1,1,1,1, 2,2,2,2]
        vals = G.T.flatten(order="F")  # flatten G by columns (matching col-major order)

        self._selection_matrix = coo_matrix(
            (vals, (rows, cols)), shape=(num_vertices, 3)
        )

        # Apply stiffness scaling
        self._selection_matrix_no_weights = self._selection_matrix.copy()
        self._selection_matrix *= self.wi

    def get_pi(self, q, frame=0):
        """
        Computes the projected deformation gradient Fhat (3x3).
        """
        v1, v2, v3, v4 = self.indices
        q1 = q[3 * v1:3 * v1 + 3]
        q2 = q[3 * v2:3 * v2 + 3]
        q3 = q[3 * v3:3 * v3 + 3]
        q4 = q[3 * v4:3 * v4 + 3]

        Ds = np.column_stack([q1 - q4, q2 - q4, q3 - q4])  # (3, 3)
        F = Ds @ self.DmInv  # (3, 3)

        U, s, Vt = np.linalg.svd(F)
        s = np.clip(s, self.sigma_min, self.sigma_max)

        if np.linalg.det(F) < 0.0:
            s[2] = -s[2]

        Fhat = U @ np.diag(s) @ Vt  # shape (3, 3)
        return Fhat

    def project_wi_SiT_pi(self, q, rhs):
        rhs += self.selection_matrix @ self.get_pi(q)

    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Returns: builds the contribution of one constrained tet vertices (self.indices)
        to the constraints related term in the LHS global matrix

        """
        # G = Ai Si
        G = np.zeros((4, 3))
        G[:3, :] = self.DmInv
        G[3, :] = -G[:3, :].sum(axis=0)  # selection matrix Si

        # Compute Gram matrix: dot products of all gradients
        K4x4 = G @ G.T  # (Ai Si)^T Ai Si, shape: (4, 4),  Ai = I

        # Kronecker product with 3x3 identity to expand each scalar to 3x3 block
        K12x12 = np.kron(K4x4, np.eye(3)) * (self.wi)

        # Convert to triplet format (i, j, val) for sparse matrix assembly
        triplets = []
        for i in range(12):
            for j in range(12):
                val = K12x12[i, j]
                if abs(val) > 1e-12:
                    triplets.append((3 * self.indices[i // 3] + i % 3, 3 * self.indices[j // 3] + j % 3, val))

        return triplets

    # def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
    #     """
    #     Args:
    #         q: current positions (from global solve)
    #         rhs: constraint projections term Sum_i wi_SiT_AiT_Bi_pi
    #
    #     Returns: computes and adds the contribution of a single tetrahedral element
    #     to the global right-hand side (RHS) vector
    #     """
    #
    #     # find the edges of the current tet position
    #     v1, v2, v3, v4 = self.indices
    #     q1 = q[3 * v1:3 * v1 + 3]
    #     q2 = q[3 * v2:3 * v2 + 3]
    #     q3 = q[3 * v3:3 * v3 + 3]
    #     q4 = q[3 * v4:3 * v4 + 3]
    #
    #     # Compute Deformation Gradient F
    #     Ds = np.column_stack([q1 - q4, q2 - q4, q3 - q4])  # deformation matrix in current configuration
    #     F = Ds @ self.DmInv
    #
    #     is_tet_inverted = np.linalg.det(F) < 0.0
    #
    #     U, s, Vt = np.linalg.svd(F)
    #     s = np.clip(s, self.sigma_min, self.sigma_max)
    #     if is_tet_inverted:
    #         s[2] = -s[2]
    #     Fhat = U @ np.diag(s) @ Vt   # pi: projection of F onto constraint manifold, shape (3, 3)
    #
    #     weight = self.wi * abs(self.V0)  # stiffness of the constraint
    #
    #     grads = self.DmInv  # (3, 3)
    #     grads_l = -np.sum(grads, axis=0)  # (3,)
    #     G = np.column_stack([grads.T, grads_l])  # A-i S_i, shape (3, 4)
    #
    #     corrections = Fhat @ G  # SiT AiT Bi pi: applying the projected deformation gradient F to the shape gradient B, shape (3, 4)
    #
    #     # Add each correction to rhs
    #     for i, idx in enumerate(self.indices):
    #         rhs[3 * idx: 3 * idx + 3] += weight * corrections[:, i]

class TetDeformationGradientConstraint(Constraint):
    """Purpose: Keeps the whole affine transformation (rotation, scaling, shearing) close to identity (rest shape)."""

    def __init__(self,id, indices, wi, positions):
        super().__init__(id, indices, wi)
        assert len(indices) == 4
        self.name = "tets_deformation_gradient"

        v1, v2, v3, v4 = indices
        p1, p2, p3, p4 = positions[v1], positions[v2], positions[v3], positions[v4]

        Dm = np.column_stack([p1 - p4, p2 - p4, p3 - p4])  # 3x3 matrix, cols: [p1-p4, p2-p4, p3-p4]
        self.DmInv = np.linalg.inv(Dm)
        self.V0 = (1.0 / 6.0) * np.linalg.det(Dm)

        # build differential operator SiT
        self.set_wi(wi * abs(self.V0))  # TODO: test
        self.build_SiT(positions.shape[0])

    def set_wi(self, value):
        self._wi = value

    def build_SiT(self, num_vertices):
        """
        Precomputes S_i^T: selection matrix transposed for this tet.
        SiT: shape (num_vertices, 4), each column corresponds to one tet vertex.
        We build a 3D gradient matrix of shape (num_vertices, 3), per-vertex.
        """
        grads = self.DmInv  # shape (3, 3)
        grads_l = -np.sum(grads, axis=0)  # shape (3,)

        G = np.column_stack([grads.T, grads_l])  # shape (3, 4)

        # Suppose: num_vertices = |V|, G is (3, 4)
        v1, v2, v3, v4 = self.indices
        rows = np.array([v1, v2, v3, v4] * 3)
        cols = np.repeat(np.arange(3), 4)  # [0,0,0,0, 1,1,1,1, 2,2,2,2]
        vals = G.T.flatten(order="F")  # flatten G by columns (matching col-major order)

        self._selection_matrix = coo_matrix(
            (vals, (rows, cols)), shape=(num_vertices, 3)
        )

        # Apply stiffness scaling
        self._selection_matrix_no_weights = self._selection_matrix.copy()
        self._selection_matrix *= self.wi

    def get_pi(self, q, frame = 0):
        """
        Computes pi = Rᵀ Gᵀ ∈ ℝ^{4×3}
        """
        v1, v2, v3, v4 = self.indices
        q1 = q[3 * v1:3 * v1 + 3]
        q2 = q[3 * v2:3 * v2 + 3]
        q3 = q[3 * v3:3 * v3 + 3]
        q4 = q[3 * v4:3 * v4 + 3]

        Ds = np.column_stack([q1 - q4, q2 - q4, q3 - q4])  # 3×3
        F = Ds @ self.DmInv  # 3×3

        U, _, Vt = svd(F)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        return  R.T  # shape (4×3)

    def project_wi_SiT_pi(self, q, rhs):
        rhs += self.selection_matrix @ self.get_pi(q)

    def tetrahedralize(self, V, F):
        tetgen_options = "pq1.2Y"  # or "pq1.414a0.01"

        TV, TT, TF = copyleft.tetgen.tetrahedralize(V, F, tetgen_options)

        # if not success:
        #     print("[ERROR] Tetrahedralization failed.")
        #     return

        TT = TT[:, ::-1]  # reverse rows
        TF = TF[:, ::-1]

        BC = barycenter(TV, TT)
        W = winding_number(V, F, BC)

        inside = (W > 0.5)
        IT = TT[inside]  # tets

        faces = boundary_facets(IT)
        F = faces[:, ::-1]

        return TV, IT, F

    # def evaluate(self, positions, masses):
    #     v1, v2, v3, v4 = self.indices
    #     p1, p2, p3, p4 = positions[v1], positions[v2], positions[v3], positions[v4]
    #
    #     Ds = np.column_stack([p1 - p4, p2 - p4, p3 - p4])
    #     Vsigned = (1. / 6.) * np.linalg.det(Ds)
    #
    #     # Check tetrahedron inversion
    #     is_V_positive = Vsigned >= 0.
    #     is_V0_positive = self.V0 >= 0.
    #     is_tet_inverted = (is_V_positive and not is_V0_positive) or (not is_V_positive and is_V0_positive)
    #
    #     # Deformation gradient
    #     F = Ds @ self.DmInv
    #     I = np.identity(3)
    #
    #     # SVD decomposition
    #     U, S, Vt = svd(F)
    #     Fhat = np.zeros((3, 3))
    #     np.fill_diagonal(Fhat, S)   # F_ii = S_i
    #
    #     V = Vt.T  # TODO check
    #
    #     if is_tet_inverted:
    #         Fhat[2, 2] *= -1
    #         U[:, 2] *= -1
    #
    #     # stress reaches maximum at 58% compression
    #     # Clamp minimum singular value to avoid instability at high compression
    #     min_singular_value = 0.577
    #     Fhat[0, 0] = max(Fhat[0, 0], min_singular_value)
    #     Fhat[1, 1] = max(Fhat[1, 1], min_singular_value)
    #     Fhat[2, 2] = max(Fhat[2, 2], min_singular_value)
    #
    #     # Material parameters
    #     young_modulus = 1_000_000_000.
    #     poisson_ratio = 0.45
    #     mu = young_modulus / (2. * (1 + poisson_ratio))
    #     lam = (young_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    #
    #     # Green strain and Piola stress
    #     Ehat = 0.5 * (Fhat.T @ Fhat - I)
    #     EhatTrace = np.trace(Ehat)
    #     Piolahat = Fhat @ ((2. * mu * Ehat) + (lam * EhatTrace * I))
    #
    #     # Energy
    #     E = U @ Ehat @ V.T
    #     Etrace = np.trace(E)
    #     psi = mu * np.sum(E * E) + 0.5 * lam * Etrace * Etrace
    #
    #     V0 = abs(self.V0)
    #     C = V0 * psi
    #
    #     return C
    #
    # def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
    #
    #     v1, v2, v3, v4 = self.indices
    #     q1 = q[3 * v1:3 * v1 + 3]
    #     q2 = q[3 * v2:3 * v2 + 3]
    #     q3 = q[3 * v3:3 * v3 + 3]
    #     q4 = q[3 * v4:3 * v4 + 3]
    #
    #     Ds = np.column_stack([q1 - q4, q2 - q4, q3 - q4])
    #     F = Ds @ self.DmInv
    #
    #     U, _, Vt = svd(F)
    #     R = U @ Vt
    #     if det(R) < 0:
    #         R[:, 2] *=-1
    #
    #
    #     # Build G (∇φ_i)
    #     G = np.zeros((4, 3))
    #     G[:3, :] = self.DmInv
    #     G[3, :] = -G[:3, :].sum(axis=0)
    #
    #
    #     # Compute each projected correction vector
    #     # result is shape (4, 3): one 3D vector per vertex
    #     corrections = G @ R.T  # shape (4, 3)
    #
    #     # Apply weighted correction to RHS
    #     weight = self.wi * abs(self.V0)
    #     for i, idx in enumerate([v1, v2, v3, v4]):
    #         rhs[3 * idx: 3 * idx + 3] += weight * corrections[i]

    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Returns: builds the contribution of one constrained tet vertices (self.indices)
        to the constraints related term in the LHS global matrix

        """
        # G = Ai Si
        G = np.zeros((4, 3))
        G[:3, :] = self.DmInv
        G[3, :] = -G[:3, :].sum(axis=0)  # grad φ₄

        # Compute Gram matrix: dot products of all gradients
        K4x4 = G @ G.T  # shape: (4, 4)

        # Kronecker product with 3x3 identity to expand each scalar to 3x3 block
        K12x12 = np.kron(K4x4, np.eye(3)) * (self.wi)

        # Convert to triplet format (i, j, val) for sparse matrix assembly
        triplets = []
        for i in range(12):
            for j in range(12):
                val = K12x12[i, j]
                if abs(val) > 1e-12:
                    triplets.append((3 * self.indices[i // 3] + i % 3, 3 * self.indices[j // 3] + j % 3, val))

        return triplets



class DeformableMesh:
    def __init__(self, positions, faces, elements=None, masses=None, mass_per_particle=2, mass_fixed_particle=1e4):

        self.floor_height = 0
        self.foolr_collision = True
        self.init_hight_shift = 1    # 3 for cloth 1 for bar

        self.init_positions = np.array(positions)  # rest positions
        self.positions = self.init_positions.copy()  # current positions

        self.positions_corrections = np.zeros_like(self.positions)
        self.faces = np.array(faces)
        self.elements = elements
        self.mass_fixed_particle = mass_fixed_particle

        n = self.positions.shape[0]
        import igl
        if self.elements is not None:
            m = igl.massmatrix(self.positions, self.elements)
        else:
            m = igl.massmatrix(self.positions, self.faces, igl.MASSMATRIX_TYPE_VORONOI)

        self.lumped_mass = np.diag(m.todense()).copy()

        # self.mass = np.ones(n)  if masses is None else np.array(masses)
        self.mass_normalization = 1/self.lumped_mass.sum()
        self.mass_per_particle = mass_per_particle
        self.mass = self.lumped_mass * self.mass_normalization * self.mass_per_particle
        self.mass_init = self.mass.copy()
        self.velocities = np.zeros_like(self.positions)

        # for position subspaces
        self.reduced_positions = None
        self.reduced_velocities = None

        self.fixed_flags = [False] * n
        self.picked_vert = [False] * n
        self.threshold_fixing_ration = 0.01
        # constraints attributes
        self.constraints = []   # list of all constrained elements each as Constraint class-instance
        self.constraints_changed = True  # assumes that no constraints are added yet
        # self.build_assembly = build_assembly

        self.has_positional_constraints = False
        self.positional_constraints = []
        self.positional_assembly_ST = None
        self.positional_stacked_p = None
        self.positional_constraints_changed = True

        self.has_verts_bending_constraints = False
        self.verts_bending_constraints = []
        self.verts_bending_assembly_ST = None
        self.verts_bending_assembly_ST_no_weights = None
        self.verts_bending_stacked_p = None
        self.verts_bending_indicies = []
        self.bending_constraints_changed = True

        self.has_edge_spring_constraints = False
        self.edge_spring_constraints = []
        self.edge_spring_assembly_ST = None
        self.edge_spring_assembly_ST_no_weights = None
        self.edge_spring_stacked_p  = None
        self.spring_constraints_changed = True


        self.has_tris_strain_constraints = False
        self.tris_strain_constraints = []
        self.tris_strain_assembly_ST = None
        self.tris_strain_assembly_ST_no_weights = None
        self.tris_strain_stacked_p = None
        self.tris_strain_constraints_changed = True


        self.has_tets_strain_constraints = False
        self.tets_strain_constraints = []
        self.tets_strain_assembly_ST = None
        self.tets_strain_assembly_ST_no_weights = None
        self.tets_strain_stacked_p = None
        self.tets_strain_constraints_changed = True


        self.has_tets_deformation_gradient_constraints = False
        self.tets_deformation_gradient_constraints = []
        self.tets_deformation_gradient_assembly_ST = None
        self.tets_deformation_gradient_assembly_ST_no_weights = None
        self.tets_deformation_gradient_stacked_p = None
        self.tets_deformation_constraints_changed = True

    def set_positions_to_initial(self):
        self.positions = self.init_positions.copy()

    def set_init_hight(self, value=1):
        self.init_hight_shift = value
        self.init_positions[:, 1] += self.init_hight_shift
        self.positions = self.init_positions.copy()  # current positions

    def reset_constraints_attributes(self):
        self.constraints = []  # list of all constrained elements each as Constraint class-instance
        # self.has_verts_positional_constraints = False
        # self.positional_constraints_assembly_ST = None

        # self.has_positional_constraints = False
        # self.positional_constraints = []
        # self.positional_assembly_ST = None
        # self.positional_stacked_p = None
        #
        # self.has_verts_bending_constraints = False
        # self.verts_bending_constraints = []
        # self.verts_bending_assembly_ST = None
        # self.verts_bending_assembly_ST_no_weights = None
        # self.verts_bending_stacked_p = None
        # self.verts_bending_indicies = []
        #
        # self.has_edge_spring_constraints = False
        # self.edge_spring_constraints = []
        # self.edge_spring_assembly_ST = None
        # self.edge_spring_assembly_ST_no_weights = None
        # self.edge_spring_stacked_p = None
        #
        # self.has_tris_strain_constraints = False
        # self.tris_strain_constraints = []
        # self.tris_strain_assembly_ST = None
        # self.tets_strain_assembly_ST_no_weights = None
        # self.tris_strain_assembly_ST_no_weights = None
        # self.tris_strain_stacked_p = None
        #
        # self.has_tets_strain_constraints = False
        # self.tets_strain_constraints = []
        # self.tets_strain_assembly_ST = None
        # self.tets_strain_stacked_p = None
        #
        # self.has_tets_deformation_gradient_constraints = False
        # self.tets_deformation_gradient_constraints = []
        # self.tets_deformation_gradient_assembly_ST = None
        # self.tets_deformation_gradient_stacked_p = None

    # def compute_sides_and_corner_indices(self):
    #     """
    #     Compute and cache the vertex indices of corners and side surfaces for each cloth side:
    #     "left", "right", "top", "bottom".
    #     """
    #     threshold_ratio = self.threshold_fixing_ration
    #     if self.positions is None:
    #         return
    #
    #     if not hasattr(self, "_cloth_corner_indices"):
    #         self._cloth_corner_indices = {}
    #
    #     positions = self.positions[:, :2]  # use only x and y
    #     x = positions[:, 0]
    #     y = positions[:, 1]
    #
    #     min_x, max_x = x.min(), x.max()
    #     min_y, max_y = y.min(), y.max()
    #     width = max_x - min_x
    #     height = max_y - min_y
    #
    #     x_thresh = threshold_ratio * width
    #     y_thresh = threshold_ratio * height
    #
    #     surface_verts = np.unique(self.faces.flatten()) if self.faces is not None else np.arange(len(x))
    #
    #     # Compute full surface vertices per side
    #     self._side_surface_verts = {}
    #
    #     for side in ["left", "right", "top", "bottom"]:
    #         if side == "left":
    #             mask = x <= min_x + x_thresh
    #         elif side == "right":
    #             mask = x >= max_x - x_thresh
    #         elif side == "bottom":
    #             mask = y <= min_y + y_thresh
    #         elif side == "top":
    #             mask = y >= max_y - y_thresh
    #
    #         self._side_surface_verts[side] = np.intersect1d(np.where(mask)[0], surface_verts)

    import numpy as np

    def compute_sides_and_corner_indices(self, plane="xy", threshold=0.02):
        """
        Compute and cache vertex indices of corners and side surfaces in a chosen plane.

        Works for cloth or bar meshes by selecting the plane that defines your 'sides':
          - plane="xy": left/right by x, bottom/top by y   (typical cloth in XY)
          - plane="xz": left/right by x, bottom/top by z   (typical bar along X with vertical Z)
          - plane="yz": left/right by y, bottom/top by z   (less common)

        Side names always map to:
          - left  : min of first axis in the plane
          - right : max of first axis in the plane
          - bottom: min of second axis in the plane
          - top   : max of second axis in the plane

        Corners are the intersections:
          - bottom_left, bottom_right, top_left, top_right

        Uses self.threshold_fixing_ration as a relative band thickness.
        """
        thr = float(getattr(self, "threshold_fixing_ration", threshold))
        if self.positions is None:
            return

        # Ensure caches exist
        if not hasattr(self, "_side_surface_verts"):
            self._side_surface_verts = {}
        if not hasattr(self, "_corner_indices"):
            self._corner_indices = {}

        # Choose axes for the plane
        plane = plane.lower()
        if plane == "xy":
            a_idx, b_idx = 0, 1
        elif plane == "xz":
            a_idx, b_idx = 0, 2
        elif plane == "yz":
            a_idx, b_idx = 1, 2
        else:
            raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

        a = self.positions[:, a_idx]
        b = self.positions[:, b_idx]

        a_min, a_max = float(a.min()), float(a.max())
        b_min, b_max = float(b.min()), float(b.max())
        a_span = max(a_max - a_min, 1e-12)
        b_span = max(b_max - b_min, 1e-12)

        a_thresh = thr * a_span
        b_thresh = thr * b_span

        # "surface_verts": if faces exist, restrict to vertices used by faces; else all
        if self.faces is not None:
            surface_verts = np.unique(self.faces.reshape(-1))
        else:
            surface_verts = np.arange(self.positions.shape[0], dtype=int)

        # Side masks in the chosen plane
        masks = {
            "left": a <= a_min + a_thresh,
            "right": a >= a_max - a_thresh,
            "bottom": b <= b_min + b_thresh,
            "top": b >= b_max - b_thresh,
        }

        # Side sets (restricted to surface_verts)
        self._side_surface_verts = {}
        for side, mask in masks.items():
            self._side_surface_verts[side] = np.intersect1d(np.where(mask)[0], surface_verts)

        # Corner sets = intersections of side sets
        bl = np.intersect1d(self._side_surface_verts["bottom"], self._side_surface_verts["left"])
        br = np.intersect1d(self._side_surface_verts["bottom"], self._side_surface_verts["right"])
        tl = np.intersect1d(self._side_surface_verts["top"], self._side_surface_verts["left"])
        tr = np.intersect1d(self._side_surface_verts["top"], self._side_surface_verts["right"])

        # If the threshold band is very thin, intersections can be empty on coarse meshes.
        # Fallback: pick closest vertex to each geometric corner in the plane.
        def closest_to_corner(a_target, b_target):
            # restrict search to surface verts for robustness
            aa = a[surface_verts]
            bb = b[surface_verts]
            d2 = (aa - a_target) ** 2 + (bb - b_target) ** 2
            return int(surface_verts[np.argmin(d2)])

        if bl.size == 0: bl = np.array([closest_to_corner(a_min, b_min)], dtype=int)
        if br.size == 0: br = np.array([closest_to_corner(a_max, b_min)], dtype=int)
        if tl.size == 0: tl = np.array([closest_to_corner(a_min, b_max)], dtype=int)
        if tr.size == 0: tr = np.array([closest_to_corner(a_max, b_max)], dtype=int)

        self._corner_indices = {
            "bottom_left": bl,
            "bottom_right": br,
            "top_left": tl,
            "top_right": tr,
        }

        # (Optional) preserve your old name if other code depends on it
        # self._cloth_corner_indices = self._corner_indices

    def get_fixed_indices(self):
        return self.fixed_flags

    def get_picked_verts(self):
        return self.picked_vert
    def toggle_picked(self, i):
        self.picked_vert[i] = not self.picked_vert[i]

    def constraints_list(self):
        return self.constraints

    def is_fixed(self, i):
        return self.fixed_flags[i]

    def fix(self, i, wi):
        self.fixed_flags[i] = True
        self.mass[i] = self.mass_fixed_particle

        # self.add_positional_constraint(i, wi, motion_type="fixed")

    def unfix(self, i):
        self.fixed_flags[i] = False
        # self.mass[i] = self.mass_init[i]
        # self.remove_positional_constraint(i)

    def toggle_fixed(self, i, mass_when_unfixed=1.0): #todo, mass when not fixed wie init
        self.fixed_flags[i] = not self.fixed_flags[i]
        self.mass[i] = self.mass_fixed_particle if self.fixed_flags[i] else self.mass_init[i]


    def fix_side_vertices(self, args, threshold = None, side="left", axis=0):
        """
        Fixes all vertices on the left or right side of the self along a given axis.

        Parameters:
            self (DeformableMesh): the deformable self
            side (str): "left" or "right"
            axis (int): axis to evaluate ('0' for x, '1' for y, '2' for z)
        """
        if self.positions is None or self.positions.shape[0] == 0:
            return

        V = self.positions.copy()
        if threshold is None:
            threshold = V[:, axis].mean()

        for i in range(V.shape[0]):
            if (side == "left" and V[i, axis] < threshold) or (side == "right" and V[i, axis] > threshold):
                self.fix(i)

    def fix_surface_side_vertices(self, wi, side="left", fix_it = True, return_target=False, threshold=0.02):
        """
        Fixes the surface vertices on the specified side of the cloth: "left", "right", "top", "bottom".
        """
        if self.positions is None or self.faces is None:
            return

        if not hasattr(self, "_side_surface_verts") or side not in self._side_surface_verts:
            self.compute_sides_and_corner_indices(threshold=threshold)

        surface_targets = self._side_surface_verts.get(side, [])
        if fix_it:
            for vi in surface_targets:
                self.fix(vi, wi)

        if return_target:
            return surface_targets
        else:
            pass

    def toggle_pick_surface_side_vertices(self, side="left", return_surface_verts=False, threshold=0.02):
        """
        Returns list of vertex indices on the chosen side surface,
        and marks them as picked for visualization.
        """
        # Ensure sides are computed
        if not hasattr(self, "_side_surface_verts") or side not in getattr(self, "_side_surface_verts", {}):
            self.compute_sides_and_corner_indices(threshold=threshold)

        surface_verts = list(self._side_surface_verts.get(side, []))

        # Mark picked (works if picked_vert is dict or boolean array)
        if hasattr(self, "picked_vert") and self.picked_vert is not None:
            for vi in surface_verts:
                try:
                    self.picked_vert[vi] = not self.picked_vert[vi]
                except Exception:
                    # If picked_vert is e.g. a dict-like but not settable by int
                    pass
        if return_surface_verts:
            return surface_verts
        else:
            pass

    def release_surface_side_vertices(self, side="left"):
        """
        Releases (unfixes) all surface vertices on the specified cloth side.
        Valid sides: "left", "right", "top", "bottom".
        """
        if not hasattr(self, "_side_surface_verts") or side not in self._side_surface_verts:
            print(
                f"[Warning] Surface side vertices not cached or side '{side}' missing. Run compute_sides_and_corner_indices() first.")
            return

        verts = self._side_surface_verts.get(side, None)
        if verts is None:
            print(f"[Warning] No cached vertices for side: {side}")
            return

        for vi in verts:
            self.unfix(vi)


    def fix_corners(self,wi, side="bottom_left", threshold=0.02):
        if not hasattr(self, "_corner_indices") or side not in self._corner_indices:
            self.compute_sides_and_corner_indices(threshold=threshold)

        indices = self._corner_indices.get(side, [])
        for vi in indices:
            self.fix(vi, wi)
            self.picked_vert[vi] = True

    def release_corners(self, side="left", threshold=0.02):
        if not hasattr(self, "bottom_left") or side not in self._corner_indices:
            self.compute_sides_and_corner_indices(threshold=threshold)

        indices = self._corner_indices.get(side, [])
        for vi in indices:
            self.unfix(vi)
            self.picked_vert[vi] = False


    def immobilize(self):
        self.velocities[:] = 0


    def triangle_area(self, tri, positions):
        """
        Compute area of a triangle given vertex indices and position array
        tri: array-like of 3 vertex indices
        positions: (N, 3) numpy array of vertex positions
        """
        v0, v1, v2 = positions[tri[0]], positions[tri[1]], positions[tri[2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    def vertex_masses(self, triangles, positions):
        """
        Compute per-vertex masses based on triangle area (1/3 each).
        Args:
            triangles: (M, 3) array of triangle indices
            positions: (N, 3) array of vertex positions
        Returns:
            v_masses: (N,) array of per-vertex mass
        """
        v_masses = np.zeros(len(positions))

        for tri in triangles:
            area = self.triangle_area(tri, positions) / 3.0
            for vi in tri:
                v_masses[vi] += area

        # Apply minimum mass
        v_masses[v_masses < 1e-7] = 1e-7
        return v_masses
    #
    # def add_constraint(self, constraint):
    #     assert isinstance(constraint, Constraint)
    #     self.constraints.append(constraint)

    def clear_constraints(self):
        self.constraints.clear()

    def count_edges(self, F):
        # Extract all edges (unordered pairs)
        edges = np.vstack([
            F[:, [0, 1]],
            F[:, [1, 2]],
            F[:, [2, 0]]
        ])

        # Sort each edge so (i, j) and (j, i) are treated the same
        edges = np.sort(edges, axis=1)

        # Get unique edges
        unique_edges = np.unique(edges, axis=0)

        return len(unique_edges)

    def vertex_star(self):
        """Build the list of Edge objects forming the 1-ring neighborhood around vertex v."""
        vertex_stars = [[] for _ in range(self.positions.shape[0])]

        num_triangles = self.faces.shape[0]
        for t in range(num_triangles):
            tri = self.faces[t]
            for v in range(3):
                vInd = tri[v]
                for ov in range(3):
                    if v == ov:
                        continue
                    nbVInd = tri[ov]
                    third_vertex = tri[3 - (v + ov)]
                    found = False

                    for edge in vertex_stars[vInd]:
                        if edge.v2 == nbVInd:
                            edge.t2 = t
                            edge.vOtherT2 = third_vertex
                            found = True
                            break

                    if not found:
                        edge = Edge(
                            v2=nbVInd,
                            vOtherT1=third_vertex,
                            t1=t,
                            vOtherT2=-1,
                            t2=-1
                        )
                        vertex_stars[vInd].append(edge)

        return vertex_stars

    def add_positional_constraint(self, vi, wi=1e9, motion_type='fixed', frames_series=None, frame_reset=0):
        self.has_positional_constraints = True
        c = PositionalConstraint(vi, [vi], wi * self.mass_normalization , self.positions, motion_type, frames_series, frame_reset)
        # self.constraints.append(c)

        # build assembly
        self.positional_constraints.append(c)
        if self.positional_assembly_ST is None:
            self.positional_assembly_ST= c.selection_matrix  # each (|V|, 2)
        else:
            col = self.positional_assembly_ST.shape[1]
            self.positional_assembly_ST.resize((self.positional_assembly_ST.shape[0], col+1))
            self.positional_assembly_ST[:, -1] = c.selection_matrix  # each (|V|, 2)

        # triggers the solver to re-prepare the global matrix and update selection matrices
        self.constraints_changed = True
        self.positional_constraints_changed = True

    def remove_positional_constraint(self, vi):

        instance = [c for c in self.positional_constraints if c.indices[0] == vi]
        self.positional_constraints = [
            c for c in self.positional_constraints if c.indices[0] != vi
        ]
        self.constraints = [c for c in self.constraints if not (isinstance(c, PositionalConstraint) and c.indices[0] == vi)]

        if len(self.positional_constraints) == 0:
            self.has_positional_constraints = False

            self.positional_assembly_ST = delete_matching_column(
                self.positional_assembly_ST,
                instance[0].selection_matrix
            )

        # triggers the solver to re-prepare the global matrix and update selection matrices
        self.constraints_changed = True
        self.positional_constraints_changed = True



    def add_vertex_bending_constraint(self, wi=1e6, samples=None):

        self.has_verts_bending_constraints = True

        voronoi_area = self.vertex_masses(self.faces, self.positions)
        # compute vertices stars for all verts
        vertex_stars = self.vertex_star()

        nV = self.positions.shape[0]

        # Build all constraints first
        if samples is not None:
            verts_to_process = samples
        else:
            verts_to_process = range(nV)

        for v in verts_to_process:
            star = vertex_stars[v]
            if not star:
                continue
            # skip boundary vertices (edges missing t2)
            if any(e.t2 < 0 for e in star):
                continue

            c = VertBendingConstraint(v, v, wi * self.mass_normalization , star, voronoi_area[v], self.positions, self.faces)
            # self.constraints.append(c)
            self.verts_bending_constraints.append(c)
            self.verts_bending_indicies.append(v)

        # Now that constraints are known, we can assemble the global sparse matrix
        nC = len(self.verts_bending_constraints)

        rows, cols, vals = [], [], []
        rows_noW, cols_noW, vals_noW = [], [], []

        for j, c in enumerate(self.verts_bending_constraints):
            sm_w = c.selection_matrix  # (|V|, 1) COO
            sm_nw = c.selection_matrix_no_weights

            rows.extend(sm_w.row.tolist())
            cols.extend([j] * sm_w.nnz)
            vals.extend(sm_w.data.tolist())

            rows_noW.extend(sm_nw.row.tolist())
            cols_noW.extend([j] * sm_nw.nnz)
            vals_noW.extend(sm_nw.data.tolist())

        # Optional SciPy representation (for inspection)
        ST_coo = coo_matrix((vals, (rows, cols)), shape=(nV, nC))
        ST_noW_coo = coo_matrix((vals_noW, (rows_noW, cols_noW)), shape=(nV, nC))

        # Convert to torch sparse COO tensors
        self.verts_bending_assembly_ST = torch.sparse_coo_tensor(
            indices=torch.tensor([rows, cols], dtype=torch.long),
            values=torch.tensor(vals, dtype=torch.float32),
            size=(nV, nC)
        ).coalesce()

        self.verts_bending_assembly_ST_no_weights = torch.sparse_coo_tensor(
            indices=torch.tensor([rows_noW, cols_noW], dtype=torch.long),
            values=torch.tensor(vals_noW, dtype=torch.float32),
            size=(nV, nC)
        ).coalesce()

        # triggers the solver to re-prepare the global matrix and update selection matrices
        self.constraints_changed = True
        self.bending_constraints_changed = True


    def add_edge_spring_constrain(self, wi=1e6, samples=None):

        if self.elements is not None:
            E = edges(self.elements)
        else:
            E = edges(self.faces)

        self.has_edge_spring_constraints = True

        nV = self.positions.shape[0]
        nE = len(samples) if samples is not None else E.shape[0]

        # pre-allocate index and value containers
        rows, cols, vals = [], [], []
        rows_noW, cols_noW, vals_noW = [], [], []

        # --- loop over edges (or sampled edges)
        edges_to_process = samples if samples is not None else range(nE)

        for e_global, e_idx in enumerate(edges_to_process):
            elem = E[e_idx]
            e0, e1 = int(elem[0]), int(elem[1])

            # Build constraint (each has .selection_matrix in COO)
            c = EdgeSpringConstraint(e_idx, [e0, e1], wi * self.mass_normalization , self.positions)
            # self.constraints.append(c)
            self.edge_spring_constraints.append(c)

            sm_w = c.selection_matrix  # coo_matrix (|V|, 1)
            sm_nw = c.selection_matrix_no_weights

            # Append entries for this column e_global
            rows.extend(sm_w.row.tolist())
            cols.extend([e_global] * sm_w.nnz)
            vals.extend(sm_w.data.tolist())

            rows_noW.extend(sm_nw.row.tolist())
            cols_noW.extend([e_global] * sm_nw.nnz)
            vals_noW.extend(sm_nw.data.tolist())

        # --- convert directly to torch sparse tensors
        self.edge_spring_assembly_ST = torch.sparse_coo_tensor(
            indices=torch.tensor([rows, cols], dtype=torch.long),
            values=torch.tensor(vals, dtype=torch.float32),
            size=(nV, nE)
        ).coalesce()

        self.edge_spring_assembly_ST_no_weights = torch.sparse_coo_tensor(
            indices=torch.tensor([rows_noW, cols_noW], dtype=torch.long),
            values=torch.tensor(vals_noW, dtype=torch.float32),
            size=(nV, nE)
        ).coalesce()

        assert self.edge_spring_assembly_ST.shape[1] == len(self.edge_spring_constraints)

        # triggers the solver to re-prepare the global matrix and update selection matrices
        self.constraints_changed = True
        self.spring_constraints_changed = True

    def add_tri_constrain_strain(self, sigma_min, sigma_max, wi=1e6, samples=None):

        self.has_tris_strain_constraints = True
        nV = self.positions.shape[0]
        nF = len(samples) if samples is not None else self.faces.shape[0]

        rows, cols, vals = [], [], []
        rows_noW, cols_noW, vals_noW = [], [], []

        # Select faces
        faces_to_process = samples if samples is not None else range(nF)

        for f_global, f_idx in enumerate(faces_to_process):
            elem = self.faces[f_idx]

            # Build constraint
            c = TriStrainConstraint(f_idx, elem.tolist(), wi * self.mass_normalization , self.positions, sigma_min, sigma_max)
            # self.constraints.append(c)
            self.tris_strain_constraints.append(c)

            sm_w = c.selection_matrix  # (num_vertices, 2) in COO
            sm_nw = c.selection_matrix_no_weights

            # Append entries for this triangle’s 2 columns
            rows.extend(sm_w.row.tolist())
            cols.extend((2 * f_global + sm_w.col).tolist())
            vals.extend(sm_w.data.tolist())

            rows_noW.extend(sm_nw.row.tolist())
            cols_noW.extend((2 * f_global + sm_nw.col).tolist())
            vals_noW.extend(sm_nw.data.tolist())

        # --- Optional: SciPy COO for debug / inspection
        ST_coo = coo_matrix((vals, (rows, cols)), shape=(nV, 2 * nF))
        ST_noW_coo = coo_matrix((vals_noW, (rows_noW, cols_noW)), shape=(nV, 2 * nF))

        # --- Convert to torch sparse COO tensors
        self.tris_strain_assembly_ST = torch.sparse_coo_tensor(
            indices=torch.tensor([rows, cols], dtype=torch.long),
            values=torch.tensor(vals, dtype=torch.float32),
            size=(nV, 2 * nF)
        ).coalesce()

        self.tris_strain_assembly_ST_no_weights = torch.sparse_coo_tensor(
            indices=torch.tensor([rows_noW, cols_noW], dtype=torch.long),
            values=torch.tensor(vals_noW, dtype=torch.float32),
            size=(nV, 2 * nF)
        ).coalesce()
        assert self.tris_strain_assembly_ST.shape[1] == 2*len(self.tris_strain_constraints)

        # triggers the solver to re-prepare the global matrix and update selection matrices
        self.constraints_changed = True
        self.tris_strain_constraints_changed = True

    def add_tet_constrain_strain(self, sigma_min, sigma_max, wi=1e6, samples=None):
        self.has_tets_strain_constraints = True

        blocks, blocks_ = [], []
        if samples is not None:
            elems = self.elements[samples]
        else:
            elems = self.elements

        for e, elem in enumerate(elems):
            idx = samples[e] if samples is not None else e
            c = TetStrainConstraint(idx, elem.tolist(), wi * self.mass_normalization , self.positions, sigma_min, sigma_max)
            # self.constraints.append(c)
            self.tets_strain_constraints.append(c)
            i = torch.from_numpy(np.vstack([c.selection_matrix.row, c.selection_matrix.col])).long().to(device)
            v = torch.from_numpy(c.selection_matrix.data).float().to(device)
            sm_torch = torch.sparse_coo_tensor(i, v, c.selection_matrix.shape, device=device)
            blocks.append(sm_torch)  # (|V|,3)

            i_ = torch.from_numpy(np.vstack([c.selection_matrix_no_weights.row, c.selection_matrix_no_weights.col])).long().to(device)
            v_ = torch.from_numpy(c.selection_matrix_no_weights.data).float().to(device)
            smw_torch = torch.sparse_coo_tensor(i_, v_, c.selection_matrix_no_weights.shape, device=device)
            blocks_.append(smw_torch)

            # concatenate along feature axis
        self.tets_strain_assembly_ST = torch.cat(blocks, dim=1)  # (|V|,3*|T|)
        self.tets_strain_assembly_ST_no_weights = torch.cat(blocks_, dim=1)  # (|V|,3*|T|)

        # if samples is not None:
        #     self.tets_strain_assembly_ST = lil_matrix((self.positions.shape[0], 3 * len(samples)))  # (|V|,3|samples|)
        # else:
        #     self.tets_strain_assembly_ST = lil_matrix((self.positions.shape[0], 3 * self.elements.shape[0]))  # (|V|,3|T|)
        #
        # if samples is not None:
        #     for e, s in enumerate(samples):
        #         elem = self.elements[s]
        #         c = TetStrainConstraint(s, elem.tolist(), wi, self.positions, sigma_min, sigma_max)
        #         self.constraints.append(c)
        #
        #         self.tets_strain_constraints.append(c)
        #         self.tets_strain_assembly_ST[:, 3 * e: 3 * e + 3] = c.selection_matrix  # each (|V|, 1)
        # else:
        #     for e, elem in  enumerate(self.elements):
        #         c = TetStrainConstraint(e, elem.tolist(), wi, self.positions, sigma_min, sigma_max)
        #         self.constraints.append(c)
        #
        #         self.tets_strain_constraints.append(c)
        #         self.tets_strain_assembly_ST[:, 3 * e: 3 * e + 3] = c.selection_matrix  # each (|V|, 1)

        assert self.tets_strain_assembly_ST.shape[1] == 3* len(self.tets_strain_constraints)

        # triggers the solver to re-prepare the global matrix and update selection matrices
        self.constraints_changed = True
        self.tets_strain_constraints_changed = True


    def add_tet_constrain_deformation_gradient(self, wi=1e6, samples=None):
        self.has_tets_deformation_gradient_constraints = True

        blocks, blocks_ = [], []
        if samples is not None:
            elems = self.elements[samples]
        else:
            elems = self.elements

        for e, elem in enumerate(elems):
            idx = samples[e] if samples is not None else e
            c = TetDeformationGradientConstraint(idx, elem.tolist(), wi * self.mass_normalization , self.positions)
            # self.constraints.append(c)
            self.tets_deformation_gradient_constraints.append(c)
            # sm = c.selection_matrix.tocoo()
            i = torch.from_numpy(np.vstack([c.selection_matrix.row, c.selection_matrix.col])).long().to(device)
            v = torch.from_numpy(c.selection_matrix.data).float().to(device)
            sm_torch = torch.sparse_coo_tensor(i, v, c.selection_matrix.shape, device=device)
            blocks.append(sm_torch)  # (|V|,3)

            i_ = torch.from_numpy(
                np.vstack([c.selection_matrix_no_weights.row, c.selection_matrix_no_weights.col])).long().to(device)
            v_ = torch.from_numpy(c.selection_matrix_no_weights.data).float().to(device)
            smw_torch = torch.sparse_coo_tensor(i_, v_, c.selection_matrix_no_weights.shape, device=device)
            blocks_.append(smw_torch)

        # concatenate along feature axis
        self.tets_deformation_gradient_assembly_ST = torch.cat(blocks, dim=1)  # (|V|,3*|T|)
        self.tets_deformation_gradient_assembly_ST_no_weights = torch.cat(blocks_, dim=1)  # (|V|,3*|T|)


        assert self.tets_deformation_gradient_assembly_ST.shape[1] == 3* len(self.tets_deformation_gradient_constraints)

        # triggers the solver to re-prepare the global matrix and update selection matrices
        self.constraints_changed = True
        self.tets_deformation_constraints_changed = True



    def resolve_collision(self, v, pos, pos_correct):
        # Set initial correction equal to the current position
        pos_correct[v] = pos[v].copy()

        if self.foolr_collision :
            if pos[v, 1] < self.floor_height:
                self.m_collisionCorrection = True
                # Snap to the floor height
                pos[v, 1] = self.floor_height

        # Compute correction vector: - (new_pos - original_pos)
        pos_correct[v] = -(pos[v] - pos_correct[v])

    def handle_tangential_movement_and_repulsion_all_vertices(self):
        for v in range(self.num_vertices):
            correction = self.position_corrections[v]
            if np.linalg.norm(correction) > 1e-12:
                velocity = self.velocities[v]
                normal_component = np.dot(velocity, correction) * correction
                tangential_velocity = velocity - normal_component
                tangential_velocity *= (1.0 - self.friction_coeff)
                tangential_velocity += correction * self.repulsion_coeff
                self.velocities[v] = tangential_velocity

    def resolve_self_collision_fast(self, vertices, min_dist=0.001, stiffness=1.0):

        faces = self.faces
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        tree = mesh.kdtree  # fast nearest-neighbor acceleration

        def point_to_triangle_distance(p, tri):
            """
            Compute the closest point on triangle `tri` to point `p`.
            Returns: distance, closest point
            """
            a, b, c = tri
            ab = b - a
            ac = c - a
            ap = p - a

            d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
            if d1 <= 0 and d2 <= 0:
                return np.linalg.norm(p - a), a

            bp = p - b
            d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
            if d3 >= 0 and d4 <= d3:
                return np.linalg.norm(p - b), b

            vc = d1 * d4 - d3 * d2
            if vc <= 0 and d1 >= 0 and d3 <= 0:
                v = d1 / (d1 - d3)
                closest = a + v * ab
                return np.linalg.norm(p - closest), closest

            cp = p - c
            d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
            if d6 >= 0 and d5 <= d6:
                return np.linalg.norm(p - c), c

            vb = d5 * d2 - d1 * d6
            if vb <= 0 and d2 >= 0 and d6 <= 0:
                w = d2 / (d2 - d6)
                closest = a + w * ac
                return np.linalg.norm(p - closest), closest

            va = d3 * d6 - d5 * d4
            if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
                u = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                closest = b + u * (c - b)
                return np.linalg.norm(p - closest), closest

            # Inside face
            denom = 1.0 / (va + vb + vc)
            v = vb * denom
            w = vc * denom
            closest = a + ab * v + ac * w
            return np.linalg.norm(p - closest), closest

        new_vertices = vertices.copy()
        for vi, p in enumerate(vertices):
            d, face_id = tree.query(p, k=5)  # only check the k closest faces
            for fi in np.atleast_1d(face_id):
                f = faces[fi]
                if vi in f:
                    continue
                tri = vertices[f]
                dist, closest = point_to_triangle_distance(p, tri)
                if dist < min_dist and dist > 1e-8:
                    dir = (p - closest) / dist
                    new_vertices[vi] += stiffness * (min_dist - dist) * dir
        return new_vertices

    def resolve_triangle_self_collisions(self, vertices, min_dist=0.001, stiffness=0.5):
        faces = self.faces

        mesh = trimesh.Trimesh(vertices, faces, process=False)

        # Compute AABBs (min and max corners for each triangle)
        aabb_min = faces.min(axis=1)  # (n, 3)
        aabb_max = faces.max(axis=1)  # (n, 3)

        # Optionally, you can stack them as a single array
        aabbs = np.stack([aabb_min, aabb_max], axis=1)  # (n, 2, 3)        centroids = mesh.triangles_center
        updated_vertices = vertices.copy()

        # Use KDTree for fast spatial queries over triangle centroids
        from scipy.spatial import cKDTree
        centroids = mesh.triangles_center
        tree = cKDTree(centroids)

        for i, (tri_i, aabb_i) in enumerate(zip(mesh.triangles, aabbs)):
            # Find triangles with centroids nearby (within a loose radius)
            nearby = tree.query_ball_point(centroids[i], r=3 * min_dist)
            for j in nearby:
                if j <= i:
                    continue  # avoid double check
                tri_j = mesh.triangles[j]

                # Skip if triangles share any vertex (adjacent or same face)
                if len(set(faces[i]) & set(faces[j])) > 0:
                    continue

                # AABB quick rejection
                aabb_j = aabbs[j]
                if not aabbs_overlap(aabb_i, aabb_j, pad=min_dist):
                    continue

                # Check actual triangle-triangle collision
                if triangles_too_close(tri_i, tri_j, min_dist):
                    resolve_overlap(tri_i, tri_j, faces[i], faces[j], updated_vertices, stiffness, min_dist)

        return updated_vertices

def aabbs_overlap(aabb1, aabb2, pad=0.0):
    """Check if two AABBs overlap (with optional padding)."""
    return np.all(aabb1[1] + pad >= aabb2[0]) and np.all(aabb2[1] + pad >= aabb1[0])

def triangles_too_close(tri1, tri2, threshold):
    """Brute force check: any vertex of one tri is too close to the other tri."""
    for p in tri1:
        if point_triangle_distance(p, tri2)[0] < threshold:
            return True
    for p in tri2:
        if point_triangle_distance(p, tri1)[0] < threshold:
            return True
    return False

def resolve_overlap(tri1, tri2, face1, face2, vertices, stiffness, min_dist):
    """Push triangles apart based on closest points."""
    for i, pi in enumerate(tri1):
        d, closest = point_triangle_distance(pi, tri2)
        if d < min_dist and d > 1e-8:
            dir = (pi - closest) / d
            vi = face1[i]
            vertices[vi] += stiffness * (min_dist - d) * dir
    for j, pj in enumerate(tri2):
        d, closest = point_triangle_distance(pj, tri1)
        if d < min_dist and d > 1e-8:
            dir = (pj - closest) / d
            vj = face2[j]
            vertices[vj] += stiffness * (min_dist - d) * dir

def point_triangle_distance(p, tri):
    """Compute closest point and distance from point p to triangle tri."""
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = ab @ ap
    d2 = ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return np.linalg.norm(ap), a

    bp = p - b
    d3 = ab @ bp
    d4 = ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return np.linalg.norm(bp), b

    cp = p - c
    d5 = ab @ cp
    d6 = ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return np.linalg.norm(cp), c

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        proj = a + v * ab
        return np.linalg.norm(p - proj), proj

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        proj = a + w * ac
        return np.linalg.norm(p - proj), proj

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = b + w * (c - b)
        return np.linalg.norm(p - proj), proj

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    proj = a + ab * v + ac * w
    return np.linalg.norm(p - proj), proj
