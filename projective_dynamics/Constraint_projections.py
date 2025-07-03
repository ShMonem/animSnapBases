# pd/constraint.py

import numpy as np
from scipy.sparse import coo_matrix
from abc import ABC, abstractmethod
from igl import edges, boundary_facets, barycenter, winding_number, copyleft
from igl.copyleft import tetgen
from numpy.linalg import svd, det, inv

from scipy.sparse import coo_matrix
class Constraint(ABC):
    def __init__(self, indices, wi=1.0):
        self._indices = list(indices)  # list of vertex indices (ints)
        self._wi = wi                  # weight (float)

    @property
    def indices(self):
        return self._indices

    @property
    def wi(self):
        return self._wi

    def evaluate(self, positions, masses):
        # Default is zero energy; override in subclass if needed
        return 0.0

    @abstractmethod
    def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
        """
        Compute projection and update rhs.
        q: full position vector (flattened)
        rhs: numpy array to accumulate contributions
        """
        pass

    @abstractmethod
    def get_wi_SiT_AiT_Ai_Si(self):
        """
        Return list of triplets (i, j, value) to be added to system matrix.
        """
        pass

class EdgeLengthConstraint(Constraint):
    def __init__(self, indices, wi, positions):
        super().__init__(indices, wi)
        assert len(indices) == 2
        e0, e1 = indices[0], indices[1]
        self.d = np.linalg.norm(positions[e0] - positions[e1])

    def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
        vi, vj = self.indices
        p1 = q[3 * vi:3 * vi + 3]
        p2 = q[3 * vj:3 * vj + 3]
        spring = p2 - p1
        length = np.linalg.norm(spring)

        if length == 0:
            return  # Avoid divide by zero

        n = spring / length
        delta = 0.5 * (length - self.d)
        pi1 = p1 + delta * n
        pi2 = p2 - delta * n

        rhs[3 * vi:3 * vi + 3] += self.wi * 0.5 * (pi1 - pi2)
        rhs[3 * vj:3 * vj + 3] += self.wi * 0.5 * (pi2 - pi1)

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

class PositionalConstraint(Constraint):
    def __init__(self, indices, wi, positions):
        super().__init__(indices, wi)
        assert len(indices) == 1
        vi = indices[0]
        self.p0 = positions[vi].reshape(3, 1)  # Column vector

    def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
        vi = self.indices[0]
        rhs[3 * vi : 3 * vi + 3] += self.wi * self.p0.flatten()

    def get_wi_SiT_AiT_Ai_Si(self):
        vi = self.indices[0]
        triplets = [
            (3 * vi + 0, 3 * vi + 0, self.wi),
            (3 * vi + 1, 3 * vi + 1, self.wi),
            (3 * vi + 2, 3 * vi + 2, self.wi),
        ]
        return triplets

class StrainConstraint(Constraint):

    def __init__(self, indices, wi, positions, sigma_min, sigma_max):
        super().__init__(indices, wi)
        assert len(indices) == 4
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
        K4x4 = G @ G.T  # (Ai Si)^T Ai Si, shape: (4, 4)

        # Kronecker product with 3x3 identity to expand each scalar to 3x3 block
        K12x12 = np.kron(K4x4, np.eye(3)) * (self.wi * abs(self.V0))

        # Convert to triplet format (i, j, val) for sparse matrix assembly
        triplets = []
        for i in range(12):
            for j in range(12):
                val = K12x12[i, j]
                if abs(val) > 1e-12:
                    triplets.append((3 * self.indices[i // 3] + i % 3, 3 * self.indices[j // 3] + j % 3, val))

        return triplets

    def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
        """
        Args:
            q: current positions (from global solve)
            rhs: constraint projections term Sum_i wi_SiT_AiT_Bi_pi

        Returns: computes and adds the contribution of a single tetrahedral element
        to the global right-hand side (RHS) vector
        """
        v1, v2, v3, v4 = self.indices
        q1 = q[3 * v1:3 * v1 + 3]
        q2 = q[3 * v2:3 * v2 + 3]
        q3 = q[3 * v3:3 * v3 + 3]
        q4 = q[3 * v4:3 * v4 + 3]

        # Compute Deformation Gradient F
        Ds = np.column_stack([q1 - q4, q2 - q4, q3 - q4])  # deformation matrix in current configuration
        F = Ds @ self.DmInv

        is_tet_inverted = np.linalg.det(F) < 0.0

        U, s, Vt = np.linalg.svd(F)
        s = np.clip(s, self.sigma_min, self.sigma_max)
        if is_tet_inverted:
            s[2] = -s[2]
        Fhat = U @ np.diag(s) @ Vt   # pi: projection of F onto constraint manifold, shape (3, 3)

        weight = self.wi * abs(self.V0)  # stiffness of the constraint

        grads = self.DmInv  # (3, 3)
        grads_l = -np.sum(grads, axis=0)  # (3,)
        G = np.column_stack([grads.T, grads_l])  # A-i S_i, shape (3, 4)

        corrections = Fhat @ G  # SiT AiT Bi pi: applying the projected deformation gradient F to the shape gradient B, shape (3, 4)

        # Add each correction to rhs
        for i, idx in enumerate(self.indices):
            rhs[3 * idx: 3 * idx + 3] += weight * corrections[:, i]



class DeformationGradientConstraint(Constraint):
    def __init__(self, indices, wi, positions):
        super().__init__(indices, wi)
        assert len(indices) == 4

        v1, v2, v3, v4 = indices
        p1, p2, p3, p4 = positions[v1], positions[v2], positions[v3], positions[v4]

        Dm = np.column_stack([p1 - p4, p2 - p4, p3 - p4])  # 3x3 matrix, cols: [p1-p4, p2-p4, p3-p4]
        self.DmInv = np.linalg.inv(Dm)
        self.V0 = (1.0 / 6.0) * np.linalg.det(Dm)

    def evaluate(self, positions, masses):
        v1, v2, v3, v4 = self.indices
        p1, p2, p3, p4 = positions[v1], positions[v2], positions[v3], positions[v4]

        Ds = np.column_stack([p1 - p4, p2 - p4, p3 - p4])
        Vsigned = (1. / 6.) * np.linalg.det(Ds)

        # Check tetrahedron inversion
        is_V_positive = Vsigned >= 0.
        is_V0_positive = self.V0 >= 0.
        is_tet_inverted = (is_V_positive and not is_V0_positive) or (not is_V_positive and is_V0_positive)

        # Deformation gradient
        F = Ds @ self.DmInv
        I = np.identity(3)

        # SVD decomposition
        U, S, Vt = svd(F)
        Fhat = np.zeros((3, 3))
        np.fill_diagonal(Fhat, S)   # F_ii = S_i

        V = Vt.T  # TODO check

        if is_tet_inverted:
            Fhat[2, 2] *= -1
            U[:, 2] *= -1

        # stress reaches maximum at 58% compression
        # Clamp minimum singular value to avoid instability at high compression
        min_singular_value = 0.577
        Fhat[0, 0] = max(Fhat[0, 0], min_singular_value)
        Fhat[1, 1] = max(Fhat[1, 1], min_singular_value)
        Fhat[2, 2] = max(Fhat[2, 2], min_singular_value)

        # Material parameters
        young_modulus = 1_000_000_000.
        poisson_ratio = 0.45
        mu = young_modulus / (2. * (1 + poisson_ratio))
        lam = (young_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

        # Green strain and Piola stress
        Ehat = 0.5 * (Fhat.T @ Fhat - I)
        EhatTrace = np.trace(Ehat)
        Piolahat = Fhat @ ((2. * mu * Ehat) + (lam * EhatTrace * I))

        # Energy
        E = U @ Ehat @ V.T
        Etrace = np.trace(E)
        psi = mu * np.sum(E * E) + 0.5 * lam * Etrace * Etrace

        V0 = abs(self.V0)
        C = V0 * psi

        return C

    def project_wi_SiT_AiT_Bi_pi(self, q, rhs):

        # TODO : change symbolic to compact form
        v1, v2, v3, v4 = self.indices
        q1 = q[3 * v1:3 * v1 + 3]
        q2 = q[3 * v2:3 * v2 + 3]
        q3 = q[3 * v3:3 * v3 + 3]
        q4 = q[3 * v4:3 * v4 + 3]

        Ds = np.column_stack([q1 - q4, q2 - q4, q3 - q4])
        F = Ds @ self.DmInv

        U, _, Vt = svd(F)
        R = U @ Vt
        if det(R) < 0:
            R[:, 2] *=-1

        p1 = R[0, 0]
        p2 = R[1, 0]
        p3 = R[2, 0]
        p4 = R[0, 1]
        p5 = R[1, 1]
        p6 = R[2, 1]
        p7 = R[0, 2]
        p8 = R[1, 2]
        p9 = R[2, 2]

        d11, d12, d13 = self.DmInv[0]
        d21, d22, d23 = self.DmInv[1]
        d31, d32, d33 = self.DmInv[2]

        _d11_d21_d31 = -d11 - d21 - d31
        _d12_d22_d32 = -d12 - d22 - d32
        _d13_d23_d33 = -d13 - d23 - d33

        bi0 = (d11 * p1) + (d12 * p4) + (d13 * p7)
        bi1 = (d11 * p2) + (d12 * p5) + (d13 * p8)
        bi2 = (d11 * p3) + (d12 * p6) + (d13 * p9)

        bj0 = (d21 * p1) + (d22 * p4) + (d23 * p7)
        bj1 = (d21 * p2) + (d22 * p5) + (d23 * p8)
        bj2 = (d21 * p3) + (d22 * p6) + (d23 * p9)

        bk0 = (d31 * p1) + (d32 * p4) + (d33 * p7)
        bk1 = (d31 * p2) + (d32 * p5) + (d33 * p8)
        bk2 = (d31 * p3) + (d32 * p6) + (d33 * p9)

        bl0 = p1 * _d11_d21_d31 + p4 * _d12_d22_d32 + p7 * _d13_d23_d33
        bl1 = p2 * _d11_d21_d31 + p5 * _d12_d22_d32 + p8 * _d13_d23_d33
        bl2 = p3 * _d11_d21_d31 + p6 * _d12_d22_d32 + p9 * _d13_d23_d33

        weight = self.wi * abs(self.V0)  # stiffness of the constraint

        print([[bi0, bi1, bi2],[bj0, bj1, bj2],[bk0, bk1, bk2], [bl0, bl1, bl2]])
        rhs[3 * v1 + 0] += weight * bi0
        rhs[3 * v1 + 1] += weight * bi1
        rhs[3 * v1 + 2] += weight * bi2

        rhs[3 * v2 + 0] += weight * bj0
        rhs[3 * v2 + 1] += weight * bj1
        rhs[3 * v2 + 2] += weight * bj2

        rhs[3 * v3 + 0] += weight * bk0
        rhs[3 * v3 + 1] += weight * bk1
        rhs[3 * v3 + 2] += weight * bk2

        rhs[3 * v4 + 0] += weight * bl0
        rhs[3 * v4 + 1] += weight * bl1
        rhs[3 * v4 + 2] += weight * bl2

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
        K12x12 = np.kron(K4x4, np.eye(3)) * (self.wi * abs(self.V0))

        # Convert to triplet format (i, j, val) for sparse matrix assembly
        triplets = []
        for i in range(12):
            for j in range(12):
                val = K12x12[i, j]
                if abs(val) > 1e-12:
                    triplets.append((3 * self.indices[i // 3] + i % 3, 3 * self.indices[j // 3] + j % 3, val))

        return triplets


class DeformableMesh:
    def __init__(self, positions, faces, elements=None, masses=None):
        self.init_positions = np.array(positions)  # rest positions
        self.positions = np.array(positions)   # current positions
        self.faces = np.array(faces)
        self.elements = np.array(elements) if elements is not None else np.empty((0, 4), dtype=int)

        n = self.positions.shape[0]
        self.mass = np.ones(n) if masses is None else np.array(masses)
        self.velocities = np.zeros_like(self.positions)
        self.constraints = []
        self.fixed_flags = [False] * n


    def get_fixed_indices(self):
        return self.fixed_flags

    def constraints_list(self):
        return self.constraints

    def is_fixed(self, i):
        return self.fixed_flags[i]

    def fix(self, i):
        self.fixed_flags[i] = True
        self.mass[i] = 1e10

    def unfix(self, i, mass):
        self.fixed_flags[i] = False
        self.mass[i] = mass

    def toggle_fixed(self, i, mass_when_unfixed=1.0):
        self.fixed_flags[i] = not self.fixed_flags[i]
        self.mass[i] = 1e10 if self.fixed_flags[i] else mass_when_unfixed

    def fix_side_vertices(self, physics_params, threshold = None, side="left", axis=0):
        """
        Fixes all vertices on the left or right side of the self along a given axis.

        Parameters:
            self (DeformableMesh): the deformable self
            side (str): "left" or "right"
            axis (int): axis to evaluate ('0' for x, '1' for y, '2' for z)
        """
        if self.positions is None or self.positions.shape[0] == 0:
            return

        V = self.positions
        if threshold is None:
            threshold = V[:, axis].mean()

        for i in range(V.shape[0]):
            if (side == "left" and V[i, axis] < threshold) or (side == "right" and V[i, axis] > threshold):
                self.fix(i)
                self.add_positional_constraint(i, physics_params["positional_constraint_wi"])

    def fix_surface_side_vertices(self, physics_params, side="left"):
        if self.positions is None or self.faces is None:
            return

        coords = self.positions[:, 0]  # x-coordinates
        min_x, max_x = coords.min(), coords.max()
        threshold = (max_x - min_x) * 0.1

        if side == "left":
            target_indices = np.where(coords <= min_x + threshold)[0]
        elif side == "right":
            target_indices = np.where(coords >= max_x - threshold)[0]
        else:
            return

        surface_verts = np.unique(self.faces.flatten())
        surface_targets = np.intersect1d(target_indices, surface_verts)

        for vi in surface_targets:
            # self.fixed_flags[vi] = True
            self.fix(vi)
            self.add_positional_constraint(vi, physics_params["positional_constraint_wi"])

    def immobilize(self):
        self.velocities[:] = 0

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
        IT = TT[inside]

        G = boundary_facets(IT)
        G = G[:, ::-1]

        self.positions = TV
        self.init_positions = TV.copy()
        self.elements = IT
        self.faces = G
        self.constraints.clear()

    def add_constraint(self, constraint):
        assert isinstance(constraint, Constraint)
        self.constraints.append(constraint)

    def clear_constraints(self):
        self.constraints.clear()

    def constrain_edge_lengths(self, wi=1e6):

        if not self.elements.shape[0]==0:
            E = edges(self.elements)
        else:
            E = edges(self.faces)

        for e in E:
            e0, e1 = e[0], e[1]
            c = EdgeLengthConstraint([e0, e1], wi, self.init_positions)
            self.constraints.append(c)

    def add_positional_constraint(self, vi, wi=1e9):
        c = PositionalConstraint([vi], wi, self.init_positions)
        self.constraints.append(c)

    def constrain_deformation_gradient(self, wi=1e6):
        for elem in self.elements:
            c = DeformationGradientConstraint(elem.tolist(), wi, self.init_positions)
            self.constraints.append(c)

    def constrain_strain(self, sigma_min, sigma_max, wi=1e6):
        for elem in self.elements:
            c = StrainConstraint(elem.tolist(), wi, self.init_positions, sigma_min, sigma_max)
            self.constraints.append(c)
