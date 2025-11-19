# constraint_projections_torch.py
import torch
from torch import Tensor
from abc import ABC, abstractmethod

# ============================================================
# Device configuration
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def to_device(x):
    """Convert numpy array or tensor to torch tensor on correct device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    import numpy as np
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)
    raise TypeError(f"Unsupported type {type(x)}")

def torch_svd_clamped(F, sigma_min=0.5, sigma_max=1.5):
    """Performs batched SVD with singular value clamping."""
    U, S, Vh = torch.linalg.svd(F)
    S = torch.clamp(S, sigma_min, sigma_max)
    return U @ torch.diag_embed(S) @ Vh

# ============================================================
# Base constraint class (GPU-ready)
# ============================================================
class Constraint(ABC):
    def __init__(self, id, indices, wi=1.0):
        self._indices = indices
        self._wi = torch.tensor(wi, dtype=dtype, device=device)
        self._id = id
        self._selection_matrix = None

    @property
    def indices(self):
        return self._indices

    @property
    def wi(self):
        return self._wi

    @abstractmethod
    def build_SiT(self, num_vertices: int):
        pass

    @abstractmethod
    def get_pi(self, q: torch.Tensor, frame=0):
        pass

    @abstractmethod
    def project_wi_SiT_pi(self, q: torch.Tensor, rhs: torch.Tensor):
        pass

    @abstractmethod
    def get_wi_SiT_AiT_Ai_Si(self):
        pass

class PositionalConstraint(Constraint):
    """
    GPU-accelerated positional constraint.
    Fixes or prescribes vertex motion.
    """

    def __init__(self, id, indices, wi, positions, motion_type='fixed', frame_shift=None, frame_reset=0):
        super().__init__(id, indices, wi)
        self.name = "positional_torch"
        assert len(indices) == 1

        vi = indices[0]
        pos = to_device(positions)
        self.p0 = pos[vi].clone()  # rest position (3,)
        self.type = motion_type
        self.frame_shift = frame_shift
        self.frame_reset = frame_reset

        self.build_SiT(pos.shape[0])

    # --------------------------------------------------------
    def build_SiT(self, num_vertices: int):
        """Builds selection matrix (|V| x 1) with only one nonzero entry."""
        sel = torch.zeros((num_vertices, 1), dtype=dtype, device=device)
        sel[self.indices[0], 0] = self.wi
        self._selection_matrix = sel

    # --------------------------------------------------------
    def get_pi(self, q: torch.Tensor, frame=0) -> torch.Tensor:
        """
        Returns target position (1x3) depending on motion type.
        """
        if self.type == 'fixed':
            return self.p0.view(1, 3)

        elif self.type == 'user_defined':
            if self.frame_shift is None:
                raise ValueError("PositionalConstraintTorch: frame_shift required for 'user_defined' type.")
            shift_idx = max(frame - self.frame_reset, 0)
            shift = torch.tensor(self.frame_shift[shift_idx], dtype=dtype, device=device)
            return (self.p0 + shift).view(1, 3)

        else:
            raise ValueError(f"Unknown motion type '{self.type}' in PositionalConstraintTorch.")

    # --------------------------------------------------------
    def project_wi_SiT_pi(self, q: torch.Tensor, rhs: torch.Tensor):
        """rhs += S_i^T @ pi"""
        rhs.add_(self._selection_matrix @ self.get_pi(q))

    # --------------------------------------------------------
    def get_wi_SiT_AiT_Ai_Si(self):
        """Returns diagonal triplets for LHS assembly."""
        vi = self.indices[0]
        w = float(self.wi.item() if isinstance(self.wi, torch.Tensor) else self.wi)
        return [
            (3 * vi + 0, 3 * vi + 0, w),
            (3 * vi + 1, 3 * vi + 1, w),
            (3 * vi + 2, 3 * vi + 2, w),
        ]

class VertBendingConstraint(Constraint):
    """
    GPU-accelerated vertex bending constraint.
    Uses cotangent weights and mean curvature to resist folding.
    """

    def __init__(
        self, id, v_ind, wi, vertex_star, voronoi_area, positions, triangles,
        prevent_bending_flips=True, flat_bending=False
    ):
        super().__init__(id, [v_ind], wi * voronoi_area)
        self.name = "verts_bending_torch"
        self.v_ind = v_ind
        self.prevent_bending_flips = prevent_bending_flips
        self.flat_bending = flat_bending

        self.positions = to_device(positions)
        self.triangles = torch.tensor(triangles, dtype=torch.long, device=device)
        self.vertex_star = vertex_star
        self.voronoi_area = torch.tensor(voronoi_area, dtype=dtype, device=device)

        self.build_SiT(self.positions.shape[0])

    # --------------------------------------------------------
    def build_SiT(self, num_vertices: int):
        pos = self.positions
        v = self.v_ind
        vertex_star = self.vertex_star

        cotan_weights = []
        triangles_seen = set()
        tri_list = []

        def compute_angle(a, b, c):
            """Compute angle at vertex b between edges (a-b) and (c-b)."""
            u = a - b
            v = c - b
            cos_angle = torch.clamp(torch.dot(u, v) /
                                    ((torch.norm(u) * torch.norm(v)) + 1e-12),
                                    -1.0, 1.0)
            return torch.acos(cos_angle)

        # --- Compute cotangent weights
        for edge in vertex_star:
            p0 = pos[v]
            p2 = pos[edge.v2]
            p1 = pos[edge.vOtherT1]

            # angle1 at p0 (vertex v)
            angle1 = compute_angle(p1, p0, p2)
            cot = 0.5 / (torch.tan(angle1) + 1e-12)

            if edge.t2 >= 0:
                p1_2 = pos[edge.vOtherT2]
                angle2 = compute_angle(p1_2, p0, p2)
                cot += 0.5 / (torch.tan(angle2) + 1e-12)

            cotan_weights.append(cot.item() / float(self.voronoi_area))

            for t in [edge.t1, edge.t2]:
                if t >= 0 and t not in triangles_seen:
                    tri_list.append(self.triangles[t])
                    triangles_seen.add(t)

        self.cotan_weights = torch.tensor(cotan_weights, dtype=dtype, device=device)

        # --- Compute rest mean curvature vector
        mean_curv = torch.zeros(3, dtype=dtype, device=device)
        for edge, w in zip(vertex_star, self.cotan_weights):
            mean_curv += (pos[v] - pos[edge.v2]) * w
        self.rest_mean_curvature = (
            torch.tensor(0.0, device=device)
            if self.flat_bending
            else torch.norm(mean_curv)
        )

        # --- Average normal
        if tri_list:
            tris = torch.stack([pos[t] for t in tri_list])  # (T,3,3)
            normals = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # (T,3)
            norms = torch.norm(normals, dim=1, keepdim=True) + 1e-12
            normals = normals / norms
            self.tri_normal = normals.mean(dim=0)
        else:
            self.tri_normal = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

        self.dot_with_normal = torch.dot(self.tri_normal, mean_curv)

        # --- Sparse selection matrix
        selection = torch.zeros((num_vertices, 1), dtype=dtype, device=device)
        selection[v, 0] = torch.sum(self.cotan_weights)
        for edge, w in zip(vertex_star, self.cotan_weights):
            selection[edge.v2, 0] = -w
        self._selection_matrix = selection * self.wi

    # --------------------------------------------------------
    def get_pi(self, q: torch.Tensor, frame=0) -> torch.Tensor:
        """Computes projected mean curvature correction vector. Returns (1,3)."""
        if q.ndim == 1:
            q = q.view(-1, 3)
        v = self.v_ind
        star_sum = torch.zeros(3, dtype=dtype, device=device)

        for edge, w in zip(self.vertex_star, self.cotan_weights):
            star_sum += (q[v] - q[edge.v2]) * w

        norm_star = torch.norm(star_sum)
        if norm_star < 1e-10:
            correction = self.tri_normal * self.rest_mean_curvature
        else:
            correction = star_sum * (self.rest_mean_curvature / norm_star)

        if self.prevent_bending_flips:
            dot = torch.dot(self.tri_normal, correction)
            if norm_star > 1e-5 and dot * self.dot_with_normal < 0:
                correction *= -1

        return correction.view(1, 3)

    # --------------------------------------------------------
    def project_wi_SiT_pi(self, q: torch.Tensor, rhs: torch.Tensor):
        """rhs += S_i^T @ pi"""
        rhs.add_(self._selection_matrix @ self.get_pi(q))

    # --------------------------------------------------------
    def get_wi_SiT_AiT_Ai_Si(self):
        """Compute contribution to global LHS matrix (triplet form)."""
        S = self._selection_matrix
        K = (S.T @ S) * self.wi
        triplets = []
        rows, cols = torch.nonzero(K, as_tuple=True)
        vals = K[rows, cols]
        for r, c, val in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            if abs(val) > 1e-12:
                for k in range(3):
                    triplets.append((3 * r + k, 3 * c + k, val))
        return triplets

class EdgeSpringConstraint(Constraint):
    """
    GPU-accelerated edge spring constraint.
    Maintains edge length near rest length d using a symmetric projection.
    """

    def __init__(self, id, indices, wi, positions):
        super().__init__(id, indices, wi)
        assert len(indices) == 2
        self.name = "edge_spring_torch"

        pos = to_device(positions)
        v0, v1 = indices
        self.d = torch.norm(pos[v0] - pos[v1])  # rest length (scalar)

        self.build_SiT(pos.shape[0])

    # --------------------------------------------------------
    def build_SiT(self, num_vertices: int):
        """
        Builds S_i^T of shape (|V| x 1) with entries [-wi, +wi] on the two vertices.
        """
        self._selection_matrix = torch.zeros((num_vertices, 1), dtype=dtype, device=device)
        i, j = self.indices
        self._selection_matrix[i, 0] = -self.wi
        self._selection_matrix[j, 0] =  self.wi

    # --------------------------------------------------------
    def get_pi(self, q: torch.Tensor, frame=0) -> torch.Tensor:
        """
        Returns the 1x3 projection vector (same as NumPy version’s (pi2 - pi1)/2).
        q can be (3N,) or (N,3).
        """
        if q.ndim == 1:
            q = q.view(-1, 3)
        i, j = self.indices
        p1, p2 = q[i], q[j]

        spring = p2 - p1
        length = torch.norm(spring)

        if length < 1e-12:
            # Degenerate edge: no update (safe zero)
            self._pi = torch.zeros(3, dtype=dtype, device=device)
            return self._pi.view(1, 3)

        normalized = spring / length
        delta = 0.5 * (length - self.d)
        pi1 = p1 + delta * normalized
        pi2 = p2 - delta * normalized

        self._pi = 0.5 * (pi2 - pi1)  # (3,)
        return self._pi.view(1, 3)    # (1,3) to match S_i^T @ pi

    # --------------------------------------------------------
    def project_wi_SiT_pi(self, q: torch.Tensor, rhs: torch.Tensor):
        """
        rhs += S_i^T @ pi, where S_i^T: (|V|x1), pi: (1x3) → (|V|x3)
        """
        rhs.add_(self._selection_matrix @ self.get_pi(q))

    # --------------------------------------------------------
    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Triplets for global LHS block. Matches the NumPy version:
        w = wi * 0.5 on the 3x3 diagonal blocks; off-diagonals are -w.
        """
        i, j = self.indices
        w = float(self.wi.item() if isinstance(self.wi, torch.Tensor) else self.wi) * 0.5

        triplets = []
        for k in range(3):
            # diag blocks
            triplets.append((3 * i + k, 3 * i + k,  w))
            triplets.append((3 * j + k, 3 * j + k,  w))
            # off-diagonal blocks
            triplets.append((3 * i + k, 3 * j + k, -w))
            triplets.append((3 * j + k, 3 * i + k, -w))
        return triplets

class TriStrainConstraint(Constraint):
    """
    GPU-accelerated triangle strain constraint.
    Clamps the in-plane singular values of F_2D to [sigma_min, sigma_max].
    """

    def __init__(self, id, indices, wi, positions, sigma_min=0.5, sigma_max=1.5):
        super().__init__(id, indices, wi)
        assert len(indices) == 3
        self.name = "tris_strain_torch"
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        pos = to_device(positions)
        v1, v2, v3 = indices
        p1, p2, p3 = pos[v1], pos[v2], pos[v3]

        # --- Build local 2D frame P (3x2)
        e1 = p2 - p1
        e2 = p3 - p1

        t1 = e1 / (torch.norm(e1) + 1e-12)                # (3,)
        e2_proj = e2 - (torch.dot(e2, t1) * t1)           # remove component along t1
        t2 = e2_proj / (torch.norm(e2_proj) + 1e-12)      # (3,)

        self.P = torch.stack([t1, t2], dim=1)             # (3,2)

        # --- Rest shape in local frame: Dm_2d (2x2) and its inverse
        rest_edges_3d = torch.stack([p2 - p1, p3 - p1], dim=1)  # (3,2)
        rest_edges_2d = self.P.T @ rest_edges_3d                # (2,2)
        self.DmInv = torch.inverse(rest_edges_2d)               # (2,2)

        # Reference area (2D)
        self.A0 = 0.5 * torch.det(rest_edges_2d)                # scalar (can be ±)
        self.build_SiT(pos.shape[0])

    # --------------------------------------------------------
    def build_SiT(self, num_vertices: int):
        """
        Builds S_i^T (|V| x 2), where each column corresponds to a 2D in-plane direction.
        """
        grads = self.DmInv.T                      # (2,2)
        grads_l = -torch.sum(grads, dim=1)        # (2,)
        G = torch.cat([grads, grads_l[:, None]], dim=1)  # (2,3)

        self._selection_matrix = torch.zeros((num_vertices, 2), dtype=dtype, device=device)
        v1, v2, v3 = self.indices
        # Column j (0..1) gets the j-th row of G distributed to the three triangle vertices
        self._selection_matrix[v1, 0] = G[0, 0]
        self._selection_matrix[v2, 0] = G[0, 1]
        self._selection_matrix[v3, 0] = G[0, 2]
        self._selection_matrix[v1, 1] = G[1, 0]
        self._selection_matrix[v2, 1] = G[1, 1]
        self._selection_matrix[v3, 1] = G[1, 2]

        self._selection_matrix *= self.wi * torch.abs(self.A0)

    # --------------------------------------------------------
    def get_pi(self, q: torch.Tensor, frame=0) -> torch.Tensor:
        """
        Returns pi in 2D strain space, shaped (2,3).
        q can be (3N,) or (N,3).
        """
        if q.ndim == 1:
            q = q.view(-1, 3)
        v1, v2, v3 = self.indices
        q1, q2, q3 = q[v1], q[v2], q[v3]

        Ds = torch.stack([q2 - q1, q3 - q1], dim=1)     # (3,2)
        Ds_2d = self.P.T @ Ds                           # (2,2)

        U, S, Vh = torch.linalg.svd(Ds_2d @ self.DmInv) # 2D deformation gradient SVD
        S = torch.clamp(S, self.sigma_min, self.sigma_max)
        Fhat_2d = U @ torch.diag(S) @ Vh                # (2,2)

        # Map back to 3D tangent plane, then to (2,3) as in your original API
        pi = (self.P @ Fhat_2d).T                       # (2,3)
        return pi

    # --------------------------------------------------------
    def project_wi_SiT_pi(self, q: torch.Tensor, rhs: torch.Tensor):
        """
        rhs += S_i^T @ pi  where S_i^T: (|V|x2), pi: (2x3) → (|V|x3)
        """
        rhs.add_(self._selection_matrix @ self.get_pi(q))

    # --------------------------------------------------------
    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Triplets (row, col, val) for global LHS block from this triangle.
        """
        grads = self.DmInv.T                 # (2,2)
        grads_l = -torch.sum(grads, dim=1)   # (2,)
        G = torch.cat([grads.T, grads_l[:, None]], dim=1)   # (2,3)

        K3x3 = G.T @ G
        K9x9 = torch.kron(K3x3, torch.eye(3, device=device)) * (self.wi * torch.abs(self.A0))

        triplets = []
        idx = self.indices
        for i in range(9):
            for j in range(9):
                val = K9x9[i, j].item()
                if abs(val) > 1e-12:
                    row = 3 * idx[i // 3] + (i % 3)
                    col = 3 * idx[j // 3] + (j % 3)
                    triplets.append((row, col, val))
        return triplets

class TetStrainConstraint(Constraint):
    """
    GPU-accelerated version of the tetrahedral strain constraint.
    Controls element stretching/compression via clamped singular values of F.
    """

    def __init__(self, id, indices, wi, positions, sigma_min=0.5, sigma_max=1.5):
        super().__init__(id, indices, wi)
        self.name = "tets_strain_torch"
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Convert positions to torch tensor
        pos = to_device(positions)

        v1, v2, v3, v4 = indices
        p1, p2, p3, p4 = pos[v1], pos[v2], pos[v3], pos[v4]

        # Reference deformation matrix (Dm)
        Dm = torch.stack([p1 - p4, p2 - p4, p3 - p4], dim=1)  # (3,3)
        self.DmInv = torch.inverse(Dm)
        self.V0 = torch.det(Dm) / 6.0

        self.build_SiT(pos.shape[0])

    # --------------------------------------------------------
    def build_SiT(self, num_vertices: int):
        """Builds the per-element differential operator (selection matrix)."""
        grads = self.DmInv                        # (3,3)
        grads_l = -torch.sum(grads, dim=0)        # (3,)
        G = torch.cat([grads.T, grads_l.unsqueeze(1)], dim=1)  # (3,4)

        # Create dense selection matrix for this tet (|V| x 3)
        self._selection_matrix = torch.zeros((num_vertices, 3), dtype=dtype, device=device)

        v1, v2, v3, v4 = self.indices
        self._selection_matrix[v1, :] = G[:, 0]
        self._selection_matrix[v2, :] = G[:, 1]
        self._selection_matrix[v3, :] = G[:, 2]
        self._selection_matrix[v4, :] = G[:, 3]

        # Apply weighting and volume scaling
        self._selection_matrix *= self.wi * torch.abs(self.V0)

    # --------------------------------------------------------
    def get_pi(self, q: torch.Tensor, frame=0) -> torch.Tensor:
        """
        Computes projected deformation gradient F̂ (3×3) for the element.
        q: (3*N,) flattened or (N,3) positions.
        """
        if q.ndim == 1:
            q = q.view(-1, 3)
        v1, v2, v3, v4 = self.indices
        q1, q2, q3, q4 = q[v1], q[v2], q[v3], q[v4]

        Ds = torch.stack([q1 - q4, q2 - q4, q3 - q4], dim=1)  # (3,3)
        F = Ds @ self.DmInv                                  # (3,3)

        # Perform SVD and clamp singular values
        U, S, Vh = torch.linalg.svd(F)
        S_clamped = torch.clamp(S, self.sigma_min, self.sigma_max)

        # Handle inverted tets: flip last singular if det(F)<0
        if torch.det(F) < 0.0:
            S_clamped[-1] *= -1

        Fhat = U @ torch.diag(S_clamped) @ Vh
        return Fhat

    # --------------------------------------------------------
    def project_wi_SiT_pi(self, q: torch.Tensor, rhs: torch.Tensor):
        """
        Applies weighted selection matrix to projected deformation gradient.
        rhs: (N,3) tensor on device.
        """
        Fhat = self.get_pi(q)
        rhs.add_(self._selection_matrix @ Fhat)  # (|V|,3) @ (3,3) → (|V|,3)

    # --------------------------------------------------------
    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Returns contribution triplets (row, col, val) for global stiffness matrix.
        Computed purely on GPU for speed.
        """
        G = torch.zeros((4, 3), dtype=dtype, device=device)
        G[:3, :] = self.DmInv
        G[3, :] = -G[:3, :].sum(dim=0)

        K4x4 = G @ G.T                                     # (4,4)
        K12x12 = torch.kron(K4x4, torch.eye(3, device=device)) * (self.wi * torch.abs(self.V0))

        triplets = []
        for i in range(12):
            for j in range(12):
                val = K12x12[i, j].item()
                if abs(val) > 1e-12:
                    row = 3 * self.indices[i // 3] + (i % 3)
                    col = 3 * self.indices[j // 3] + (j % 3)
                    triplets.append((row, col, val))
        return triplets

class TetDeformationGradientConstraint(Constraint):
    """
    GPU-accelerated deformation-gradient constraint.
    Keeps the full affine transform (rotation-dominant) near rest by projecting F to its rotation R (polar via SVD).
    """

    def __init__(self, id, indices, wi, positions):
        super().__init__(id, indices, wi)
        self.name = "tets_deformation_gradient_torch"

        pos = to_device(positions)
        v1, v2, v3, v4 = indices
        p1, p2, p3, p4 = pos[v1], pos[v2], pos[v3], pos[v4]

        # Reference matrix Dm and cached values
        Dm = torch.stack([p1 - p4, p2 - p4, p3 - p4], dim=1)  # (3,3)
        self.DmInv = torch.inverse(Dm)
        self.V0 = torch.det(Dm) / 6.0

        self.build_SiT(pos.shape[0])

    # --------------------------------------------------------
    def build_SiT(self, num_vertices: int):
        """
        Precomputes S_i^T (|V| x 3) using the per-vertex gradients of tet shape functions.
        """
        grads = self.DmInv                  # (3,3)
        grads_l = -torch.sum(grads, dim=0)  # (3,)
        G = torch.cat([grads.T, grads_l.unsqueeze(1)], dim=1)  # (3,4)

        self._selection_matrix = torch.zeros((num_vertices, 3), dtype=dtype, device=device)
        v1, v2, v3, v4 = self.indices
        self._selection_matrix[v1, :] = G[:, 0]
        self._selection_matrix[v2, :] = G[:, 1]
        self._selection_matrix[v3, :] = G[:, 2]
        self._selection_matrix[v4, :] = G[:, 3]

        self._selection_matrix *= self.wi * torch.abs(self.V0)

    # --------------------------------------------------------
    def get_pi(self, q: torch.Tensor, frame=0) -> torch.Tensor:
        """
        Returns the rotation R (3x3) from the polar decomposition of F = Ds * DmInv.
        q may be (3N,) or (N,3).
        """
        if q.ndim == 1:
            q = q.view(-1, 3)
        v1, v2, v3, v4 = self.indices
        q1, q2, q3, q4 = q[v1], q[v2], q[v3], q[v4]

        Ds = torch.stack([q1 - q4, q2 - q4, q3 - q4], dim=1)  # (3,3)
        F = Ds @ self.DmInv                                   # (3,3)

        U, _, Vh = torch.linalg.svd(F)
        R = U @ Vh
        # Proper rotation fix (det(R)=+1)
        if torch.det(R) < 0:
            U = U.clone()
            U[:, -1] *= -1
            R = U @ Vh
        return R  # (3,3)

    # --------------------------------------------------------
    def project_wi_SiT_pi(self, q: torch.Tensor, rhs: torch.Tensor):
        """
        Apply weighted projection: rhs += S_i^T @ R
        rhs is (N,3) on device; all ops stay on GPU.
        """
        R = self.get_pi(q)                    # (3,3)
        rhs.add_(self._selection_matrix @ R)  # (|V|,3) @ (3,3) → (|V|,3)

    # --------------------------------------------------------
    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Triplets (row, col, val) for assembling the global LHS block from this tet.
        Since A_i = I, this is identical to the strain constraint’s Gram assembly.
        """
        G = torch.zeros((4, 3), dtype=dtype, device=device)
        G[:3, :] = self.DmInv
        G[3, :] = -G[:3, :].sum(dim=0)

        K4x4 = G @ G.T
        K12x12 = torch.kron(K4x4, torch.eye(3, device=device)) * (self.wi * torch.abs(self.V0))

        triplets = []
        for i in range(12):
            for j in range(12):
                val = K12x12[i, j].item()
                if abs(val) > 1e-12:
                    row = 3 * self.indices[i // 3] + (i % 3)
                    col = 3 * self.indices[j // 3] + (j % 3)
                    triplets.append((row, col, val))
        return triplets


def build_assembly(constaints_list, num_vertices, aux_size, dtype, device):
    if not constaints_list:
        return None

    rows, cols, vals = [], [], []
    for j, c in enumerate(constaints_list):
        S = c._selection_matrix.to_sparse_coo()
        idx = S.indices()
        val = S.values()
        rows.extend(idx[0].tolist())
        cols.extend((idx[1] + j*aux_size).tolist())  # offset column
        vals.extend(val.tolist())

    shape = (num_vertices, aux_size* len(constaints_list))

    device = vals.device if isinstance(vals, torch.Tensor) else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=torch.float32, device=device)

    assembly = torch.sparse_coo_tensor(indices, values, size=shape, device=device).coalesce()

    return assembly


class DeformableMesh:
    """
    GPU-accelerated deformable mesh class.
    Handles positions, masses, and constraint objects on the same device.
    """

    def __init__(self, positions, faces=None, elements=None, masses=None, device=None, dtype=torch.float32):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # Base geometry
        self.positions = torch.tensor(positions, dtype=dtype, device=self.device)
        self.init_positions = self.positions.clone()
        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device) if faces is not None else None
        self.elements = torch.tensor(elements, dtype=torch.long, device=self.device) if elements is not None else None

        self.num_vertices = self.positions.shape[0]

        # Mass vector
        if masses is None:
            self.mass = torch.ones(self.num_vertices, dtype=dtype, device=self.device)
        else:
            self.mass = torch.tensor(masses, dtype=dtype, device=self.device)
        self.mass_init = self.mass.clone()

        # Velocities and corrections
        self.velocities = torch.zeros_like(self.positions)
        self.positions_corrections = torch.zeros_like(self.positions)

        # Fixed vertex flags
        self.fixed_flags = torch.zeros(self.num_vertices, dtype=torch.bool, device=self.device)

        # Constraint containers
        self.constraints = []
        self.vertex_bending_constraints = []
        self.positional_constraints = []
        self.edge_spring_constraints = []
        self.tri_strain_constraints = []
        self.tet_strain_constraints = []
        self.tet_deformation_gradient_constraints = []

        #
        self.verts_bending_assembly_ST = None
        self.edge_spring_assembly_ST = None
        self.tris_strain_assembly_ST = None
        self.tets_strain_assembly_ST = None
        self.tets_deformation_gradient_assembly_ST = None

        self.fixed_flags = [False] * positions.shape[0]
        self.picked_vert = [False] * positions.shape[0]
        self.threshold_fixing_ration = 0.01

    # --------------------------------------------------------
    def immobilize(self):
        """Zero out velocities (useful for fixed or paused simulation)."""
        self.velocities.zero_()
    # --------------------------------------------------------
    def is_fixed(self, i):
        """True if vertex is fixed"""
        return self.fixed_flags[i]

    def fix(self, vid: int):
        """Fix a vertex by index."""
        self.fixed_flags[vid] = True
        self.mass[vid] = 1e10  # large mass to effectively fix

    def unfix(self, vid: int):
        """Release a vertex."""
        self.fixed_flags[vid] = False
        self.mass[vid] = self.mass_init[vid]

    def toggle_fixed(self, vid: int):
        self.fixed_flags[vid] = ~self.fixed_flags[vid]
        self.mass[vid] = 1e10 if self.fixed_flags[vid] else self.mass_init[vid]
    # --------------------------------------------------------
    def compute_sides_and_corner_indices(self):
        """
        Compute and cache the vertex indices of corners and side surfaces for each cloth side:
        "left", "right", "top", "bottom".
        """
        threshold_ratio = self.threshold_fixing_ration
        if self.positions is None:
            return

        if not hasattr(self, "_cloth_corner_indices"):
            self._cloth_corner_indices = {}

        positions = self.positions[:, :2]  # use only x and y
        x = positions[:, 0]
        y = positions[:, 1]

        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        width = max_x - min_x
        height = max_y - min_y

        x_thresh = threshold_ratio * width
        y_thresh = threshold_ratio * height

        import numpy as np
        surface_verts = np.unique(
            self.faces.detach().cpu().numpy().flatten()
        ) if self.faces is not None else np.arange(len(x))

        # Compute full surface vertices per side
        self._side_surface_verts = {}

        for side in ["left", "right", "top", "bottom"]:
            if side == "left":
                mask = x <= min_x + x_thresh
            elif side == "right":
                mask = x >= max_x - x_thresh
            elif side == "bottom":
                mask = y <= min_y + y_thresh
            elif side == "top":
                mask = y >= max_y - y_thresh

            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            surf_np = surface_verts.detach().cpu().numpy() if isinstance(surface_verts, torch.Tensor) else surface_verts
            self._side_surface_verts[side] = np.intersect1d(np.where(mask_np)[0], surf_np)

    def get_fixed_indices(self):
        return self.fixed_flags

    def get_picked_verts(self):
        return self.picked_vert
    def toggle_picked(self, i):
        self.picked_vert[i] = not self.picked_vert[i]

    def fix_surface_side_vertices(self, side="left", fix_it = True, return_target=False):
        """
        Fixes the surface vertices on the specified side of the cloth: "left", "right", "top", "bottom".
        """
        if self.positions is None or self.faces is None:
            return

        if not hasattr(self, "_side_surface_verts") or side not in self._side_surface_verts:
            self.compute_sides_and_corner_indices()

        surface_targets = self._side_surface_verts.get(side, [])
        if fix_it:
            for vi in surface_targets:
                self.fix(vi)

        if return_target:
            return surface_targets
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

    # --------------------------------------------------------
    def add_constraint(self, constraint):
        """Add any constraint object (Torch variant)."""
        self.constraints.append(constraint)
    # --------------------------------------------------------
    def clear_constraints(self):
        """Delete any constraint objects"""
        """Remove all constraints from the model."""
        self.constraints.clear()
        self.positional_constraints.clear()
        self.edge_spring_constraints.clear()
        self.tri_strain_constraints.clear()
        self.tet_strain_constraints.clear()
        self.tet_deformation_gradient_constraints.clear()
    # --------------------------------------------------------
    def add_positional_constraint(self, vid, wi=1e9, motion_type='fixed', frame_shift=None, frame_reset=0):
        """Attach positional constraint to vertex vid."""
        c = PositionalConstraint(vid, [vid], wi, self.positions,
                                      motion_type, frame_shift, frame_reset)
        self.positional_constraints.append(c)
        self.constraints.append(c)

    def count_edges(self, faces=None):
        import numpy as np

        """
        Count the number of unique edges in the mesh (useful for debugging/Polyscope display).
        Works for triangle or tet meshes.
        """
        if faces is None and self.faces is None:
            raise ValueError("No faces provided to count_edges()")
        if faces is None:
            faces = self.faces

        # Ensure data is on CPU and numpy
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()

        # Extract unique undirected edges
        edges = np.concatenate([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ], axis=0)
        edges = np.sort(edges, axis=1)
        unique_edges = np.unique(edges, axis=0)
        return len(unique_edges)

    # --------------------------------------------------------
    def add_edge_spring_constraint(self, wi=1e6, samples=None):
        """Add edge-spring constraints for each edge (faces or elements)."""
        if self.elements is not None and len(self.elements) > 0:
            from igl import edges
            E = edges(self.elements.cpu().numpy())
        elif self.faces is not None:
            from igl import edges
            E = edges(self.faces.cpu().numpy())
        else:
            raise ValueError("No faces or elements defined for edge springs.")

        if samples is None:
            samples = range(len(E))

        for s in samples:
            e0, e1 = E[s]
            c = EdgeSpringConstraint(s, [int(e0), int(e1)], wi, self.positions.cpu().numpy())
            self.edge_spring_constraints.append(c)
            self.constraints.append(c)


    # --------------------------------------------------------
    def add_tri_strain_constraint(self, sigma_min, sigma_max, wi=1e6, samples=None):
        """Add strain constraints for surface triangles."""
        if self.faces is None:
            raise ValueError("No triangle faces provided.")
        if samples is None:
            samples = range(self.faces.shape[0])

        for s in samples:
            elem = self.faces[s].tolist()
            c = TriStrainConstraint(s, elem, wi, self.positions.cpu().numpy(),
                                         sigma_min, sigma_max)
            self.tri_strain_constraints.append(c)
            self.constraints.append(c)

    # --------------------------------------------------------
    def add_tet_strain_constraint(self, sigma_min, sigma_max, wi=1e6, samples=None):
        """Add volumetric strain constraints for tetrahedra."""
        if self.elements is None:
            raise ValueError("No tetrahedral elements provided.")
        if samples is None:
            samples = range(self.elements.shape[0])

        for s in samples:
            elem = self.elements[s].tolist()
            c = TetStrainConstraint(s, elem, wi, self.positions.cpu().numpy(),
                                         sigma_min, sigma_max)
            self.tet_strain_constraints.append(c)
            self.constraints.append(c)


    # --------------------------------------------------------
    def add_tet_deformation_gradient_constraint(self, wi=1e6, samples=None):
        """Add full deformation gradient constraints for tetrahedra."""
        if self.elements is None:
            raise ValueError("No tetrahedral elements provided.")
        if samples is None:
            samples = range(self.elements.shape[0])

        for s in samples:
            elem = self.elements[s].tolist()
            c = TetDeformationGradientConstraint(s, elem, wi, self.positions.cpu().numpy())
            self.tet_deformation_gradient_constraints.append(c)
            self.constraints.append(c)

        self.tets_deformation_gradient_assembly_ST = build_assembly(self.tet_deformation_gradient_constraints, self.positions.cpu().numpy().shape[0], 3, dtype, device)

    # --------------------------------------------------------
    def build_rhs(self, q: Tensor) -> Tensor:
        """
        Accumulate constraint projections into RHS.
        Equivalent to ∑ w_i S_i^T p_i.
        """
        rhs = torch.zeros_like(self.positions)
        for c in self.constraints:
            c.project_wi_SiT_pi(q, rhs)
        return rhs

    # --------------------------------------------------------
    def assemble_triplets(self):
        """
        Collects all (i,j,value) triplets from constraints for building the system matrix.
        """
        triplets = []
        for c in self.constraints:
            triplets.extend(c.get_wi_SiT_AiT_Ai_Si())
        return triplets
