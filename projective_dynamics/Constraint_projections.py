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
    def get_wi_SiT_AiT_Ai_Si(self, positions, masses):
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

    def get_wi_SiT_AiT_Ai_Si(self, positions, masses):
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

    def get_wi_SiT_AiT_Ai_Si(self, positions, masses):
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

        v1, v2, v3, v4 = indices
        p1, p2, p3, p4 = positions[v1], positions[v2], positions[v3], positions[v4]

        Dm = np.stack([(p1 - p4), (p2 - p4), (p3 - p4)], axis=1)
        self.DmInv = inv(Dm)
        self.V0 = abs(np.linalg.det(Dm)) / 6.0
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
        v1, v2, v3, v4 = self.indices
        q1 = q[3 * v1:3 * v1 + 3]
        q2 = q[3 * v2:3 * v2 + 3]
        q3 = q[3 * v3:3 * v3 + 3]
        q4 = q[3 * v4:3 * v4 + 3]

        Ds = np.stack([q1 - q4, q2 - q4, q3 - q4], axis=1)
        F = Ds @ self.DmInv

        U, s, Vt = svd(F)
        s = np.clip(s, self.sigma_min, self.sigma_max)
        if det(F) < 0:
            s[2] = -s[2]

        F_hat = U @ np.diag(s) @ Vt
        weight = self.wi * self.V0

        grads = self.DmInv
        corrections = [F_hat @ grads[:, i] for i in range(3)]
        corrections.append(-sum(corrections))

        for idx, corr in zip(self.indices, corrections):
            rhs[3 * idx:3 * idx + 3] += weight * corr

    def get_wi_SiT_AiT_Ai_Si(self, positions, masses):
        weight = self.wi * self.V0
        grads = self.DmInv

        triplets = []
        for i in range(4):
            for j in range(4):
                grad_i = grads[:, i] if i < 3 else -grads.sum(axis=1)
                grad_j = grads[:, j] if j < 3 else -grads.sum(axis=1)
                for a in range(3):
                    for b in range(3):
                        row = 3 * self.indices[i] + a
                        col = 3 * self.indices[j] + b
                        val = weight * grad_i[a] * grad_j[b]
                        triplets.append((row, col, val))
        return triplets

class DeformationGradientConstraint(Constraint):
    def __init__(self, indices, wi, positions):
        super().__init__(indices, wi)
        assert len(indices) == 4

        v1, v2, v3, v4 = indices
        p1, p2, p3, p4 = positions[v1], positions[v2], positions[v3], positions[v4]

        Dm = np.stack([(p1 - p4), (p2 - p4), (p3 - p4)], axis=1)
        self.DmInv = inv(Dm)
        self.V0 = abs(np.linalg.det(Dm)) / 6.0

    def evaluate(self, positions, masses):
        v1, v2, v3, v4 = self.indices
        p1, p2, p3, p4 = positions[v1], positions[v2], positions[v3], positions[v4]

        Ds = np.stack([p1 - p4, p2 - p4, p3 - p4], axis=1)
        F = Ds @ self.DmInv

        U, s, Vt = svd(F)
        if det(F) < 0:
            s[2] = -s[2]

        F_hat = np.diag(np.maximum(s, 0.577))
        F_hat = U @ F_hat @ Vt

        I = np.eye(3)
        E = 0.5 * (F_hat.T @ F_hat - I)
        trace_E = np.trace(E)

        young_modulus = 1e9
        poisson_ratio = 0.45
        mu = young_modulus / (2 * (1 + poisson_ratio))
        lam = (young_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

        psi = mu * np.sum(E**2) + 0.5 * lam * trace_E**2
        return self.V0 * psi

    def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
        v1, v2, v3, v4 = self.indices
        q1 = q[3 * v1:3 * v1 + 3]
        q2 = q[3 * v2:3 * v2 + 3]
        q3 = q[3 * v3:3 * v3 + 3]
        q4 = q[3 * v4:3 * v4 + 3]

        Ds = np.stack([q1 - q4, q2 - q4, q3 - q4], axis=1)
        F = Ds @ self.DmInv

        U, _, Vt = svd(F)
        R = U @ Vt
        if det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        grads = self.DmInv
        weight = self.wi * self.V0

        corrections = [R @ grads[:, i] for i in range(3)]
        corrections.append(-sum(corrections))

        for idx, corr in zip(self.indices, corrections):
            rhs[3 * idx:3 * idx + 3] += weight * corr

    def get_wi_SiT_AiT_Ai_Si(self, positions, masses):
        grads = self.DmInv
        weight = self.wi * self.V0

        triplets = []
        for i in range(4):
            for j in range(4):
                grad_i = grads[:, i] if i < 3 else -grads.sum(axis=1)
                grad_j = grads[:, j] if j < 3 else -grads.sum(axis=1)
                for a in range(3):
                    for b in range(3):
                        row = 3 * self.indices[i] + a
                        col = 3 * self.indices[j] + b
                        val = weight * grad_i[a] * grad_j[b]
                        triplets.append((row, col, val))
        return triplets

class DeformableMesh:
    def __init__(self, positions, faces, elements=None, masses=None):
        self.p0 = np.array(positions)  # rest positions
        self.p = np.array(positions)   # current positions
        self.F = np.array(faces)
        self.E = np.array(elements) if elements is not None else np.empty((0, 4), dtype=int)

        n = self.p.shape[0]
        self.m = np.ones(n) if masses is None else np.array(masses)
        self.v = np.zeros_like(self.p)
        self.constraints = []
        self.fixed_flags = [False] * n

    def positions(self):
        return self.p

    def faces(self):
        return self.F

    def elements(self):
        return self.E

    def mass(self):
        return self.m

    def velocity(self):
        return self.v

    def fixed(self):
        return self.fixed_flags

    def constraints_list(self):
        return self.constraints

    def fix(self, i):
        self.fixed_flags[i] = True
        self.m[i] = 1e10

    def unfix(self, i, mass):
        self.fixed_flags[i] = False
        self.m[i] = mass

    def toggle_fixed(self, i, mass_when_unfixed=1.0):
        self.fixed_flags[i] = not self.fixed_flags[i]
        self.m[i] = 1e10 if self.fixed_flags[i] else mass_when_unfixed

    def immobilize(self):
        self.v[:] = 0

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

        self.p = TV
        self.p0 = TV.copy()
        self.E = IT
        self.F = G
        self.constraints.clear()

    def add_constraint(self, constraint):
        assert isinstance(constraint, Constraint)
        self.constraints.append(constraint)

    def clear_constraints(self):
        self.constraints.clear()

    def constrain_edge_lengths(self, wi=1e6):
        E = edges(self.E)
        for e in E:
            e0, e1 = e[0], e[1]
            c = EdgeLengthConstraint([e0, e1], wi, self.p0)
            self.constraints.append(c)

    def add_positional_constraint(self, vi, wi=1e9):
        c = PositionalConstraint([vi], wi, self.p0)
        self.constraints.append(c)

    def constrain_deformation_gradient(self, wi=1e6):
        for elem in self.E:
            c = DeformationGradientConstraint(elem.tolist(), wi, self.p0)
            self.constraints.append(c)

    def constrain_strain(self, sigma_min, sigma_max, wi=1e6):
        for elem in self.E:
            c = StrainConstraint(elem.tolist(), wi, self.p0, sigma_min, sigma_max)
            self.constraints.append(c)
