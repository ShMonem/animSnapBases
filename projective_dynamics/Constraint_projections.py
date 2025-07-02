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
    # def __init__(self, indices, wi, positions, sigma_min, sigma_max):
    #     super().__init__(indices, wi)
    #     assert len(indices) == 4
    #
    #     v1, v2, v3, v4 = indices
    #     p1, p2, p3, p4 = positions[v1], positions[v2], positions[v3], positions[v4]
    #
    #     Dm = np.stack([(p1 - p4), (p2 - p4), (p3 - p4)], axis=1)
    #     self.DmInv = inv(Dm)
    #     self.V0 = abs(np.linalg.det(Dm)) / 6.0
    #     self.sigma_min = sigma_min
    #     self.sigma_max = sigma_max

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
        self.DmInv = np.linalg.inv(Dm)
        self.V0 = (1.0 / 6.0) * np.linalg.det(Dm)

        print("")

    def project_wi_SiT_AiT_Bi_pi(self, q, rhs):
        v1, v2, v3, v4 = self.indices
        q1 = q[3 * v1:3 * v1 + 3]
        q2 = q[3 * v2:3 * v2 + 3]
        q3 = q[3 * v3:3 * v3 + 3]
        q4 = q[3 * v4:3 * v4 + 3]

        Ds = np.column_stack([q1 - q4, q2 - q4, q3 - q4])
        F = Ds @ self.DmInv

        is_tet_inverted = np.linalg.det(F) < 0.0

        U, s, Vt = np.linalg.svd(F)
        s = np.clip(s, self.sigma_min, self.sigma_max)
        if is_tet_inverted:
            s[2] = -s[2]
        Fhat = U @ np.diag(s) @ Vt

        p1, p2, p3 = Fhat[:, 0]
        p4, p5, p6 = Fhat[:, 1]
        p7, p8, p9 = Fhat[:, 2]

        d11, d12, d13 = self.DmInv[0]
        d21, d22, d23 = self.DmInv[1]
        d31, d32, d33 = self.DmInv[2]

        _d1 = -(d11 + d21 + d31)
        _d2 = -(d12 + d22 + d32)
        _d3 = -(d13 + d23 + d33)

        weight = self.wi * abs(self.V0)

        # For each vertex, compute contribution and add to rhs
        def add_rhs(idx, coeffs):
            i0 = 3 * idx
            rhs[i0 + 0] += weight * coeffs[0]
            rhs[i0 + 1] += weight * coeffs[1]
            rhs[i0 + 2] += weight * coeffs[2]

        bi = (
            d11 * p1 + d12 * p4 + d13 * p7,
            d11 * p2 + d12 * p5 + d13 * p8,
            d11 * p3 + d12 * p6 + d13 * p9,
        )
        bj = (
            d21 * p1 + d22 * p4 + d23 * p7,
            d21 * p2 + d22 * p5 + d23 * p8,
            d21 * p3 + d22 * p6 + d23 * p9,
        )
        bk = (
            d31 * p1 + d32 * p4 + d33 * p7,
            d31 * p2 + d32 * p5 + d33 * p8,
            d31 * p3 + d32 * p6 + d33 * p9,
        )
        bl = (
            _d1 * p1 + _d2 * p4 + _d3 * p7,
            _d1 * p2 + _d2 * p5 + _d3 * p8,
            _d1 * p3 + _d2 * p6 + _d3 * p9,
        )

        add_rhs(v1, bi)
        add_rhs(v2, bj)
        add_rhs(v3, bk)
        add_rhs(v4, bl)
        print("")

    def get_wi_SiT_AiT_Ai_Si(self, positions, masses):
        v1, v2, v3, v4 = self.indices

        vi = 3 * v1
        vj = 3 * v2
        vk = 3 * v3
        vl = 3 * v4

        d11, d12, d13 = self.DmInv[0]
        d21, d22, d23 = self.DmInv[1]
        d31, d32, d33 = self.DmInv[2]

        _d11_d21_d31 = -d11 - d21 - d31
        _d12_d22_d32 = -d12 - d22 - d32
        _d13_d23_d33 = -d13 - d23 - d33

        weight = self.wi * abs(self.V0)

        # col 1
        s1_1 = d11 * d11 + d12 * d12 + d13 * d13
        s4_1 = d11 * d21 + d12 * d22 + d13 * d23
        s7_1 = d11 * d31 + d12 * d32 + d13 * d33
        s10_1 = d11 * _d11_d21_d31 + d12 * _d12_d22_d32 + d13 * _d13_d23_d33
        # col 2
        s2_2 = s1_1
        s5_2 = s4_1
        s8_2 = s7_1
        s11_2 = s10_1
        # col 3
        s3_3 = s1_1
        s6_3 = s4_1
        s9_3 = s7_1
        s12_3 = s10_1
        # col 4
        s1_4 = d11 * d21 + d12 * d22 + d13 * d23
        s4_4 = d21 * d21 + d22 * d22 + d23 * d23
        s7_4 = d31 * d21 + d32 * d22 + d33 * d23
        s10_4 = d21 * _d11_d21_d31 + d22 * _d12_d22_d32 + d23 * _d13_d23_d33
        # col 5
        s2_5 = s1_4
        s5_5 = s4_4
        s8_5 = s7_4
        s11_5 = s10_4
        # col 6
        s3_6 = s1_4
        s6_6 = s4_4
        s9_6 = s7_4
        s12_6 = s10_4
        # col 7
        s1_7 = d11 * d31 + d12 * d32 + d13 * d33
        s4_7 = d21 * d31 + d22 * d32 + d23 * d33
        s7_7 = d31 * d31 + d32 * d32 + d33 * d33
        s10_7 = _d11_d21_d31 * d31 + _d12_d22_d32 * d32 + _d13_d23_d33 * d33
        # col 8
        s2_8 = s1_7
        s5_8 = s4_7
        s8_8 = s7_7
        s11_8 = s10_7
        # col 9
        s3_9 = s1_7
        s6_9 = s4_7
        s9_9 = s7_7
        s12_9 = s10_7
        # col 10
        s1_10 = _d11_d21_d31 * d11 + _d12_d22_d32 * d12 + _d13_d23_d33 * d13
        s4_10 = _d11_d21_d31 * d21 + _d12_d22_d32 * d22 + _d13_d23_d33 * d23
        s7_10 = _d11_d21_d31 * d31 + _d12_d22_d32 * d32 + _d13_d23_d33 * d33
        s10_10 = (_d11_d21_d31) ** 2 + (_d12_d22_d32) ** 2 + (_d13_d23_d33) ** 2
        # col 11
        s2_11 = s1_10
        s5_11 = s4_10
        s8_11 = s7_10
        s11_11 = s10_10
        # col 12
        s3_12 = s1_10
        s6_12 = s4_10
        s9_12 = s7_10
        s12_12 = s10_10

        triplets = []

        row1 = vi
        row2 = vi + 1
        row3 = vi + 2
        row4 = vj
        row5 = vj + 1
        row6 = vj + 2
        row7 = vk
        row8 = vk + 1
        row9 = vk + 2
        row10 = vl
        row11 = vl + 1
        row12 = vl + 2

        col1 = row1
        col2 = row2
        col3 = row3
        col4 = row4
        col5 = row5
        col6 = row6
        col7 = row7
        col8 = row8
        col9 = row9
        col10 = row10
        col11 = row11
        col12 = row12

        # col 1
        triplets.append((row1, col1, weight * s1_1))
        triplets.append((row4, col1, weight * s4_1))
        triplets.append((row7, col1, weight * s7_1))
        triplets.append((row10, col1, weight * s10_1))
        # col 2
        triplets.append((row2, col2, weight * s2_2))
        triplets.append((row5, col2, weight * s5_2))
        triplets.append((row8, col2, weight * s8_2))
        triplets.append((row11, col2, weight * s11_2))
        # col 3
        triplets.append((row3, col3, weight * s3_3))
        triplets.append((row6, col3, weight * s6_3))
        triplets.append((row9, col3, weight * s9_3))
        triplets.append((row12, col3, weight * s12_3))
        # col 4
        triplets.append((row1, col4, weight * s1_4))
        triplets.append((row4, col4, weight * s4_4))
        triplets.append((row7, col4, weight * s7_4))
        triplets.append((row10, col4, weight * s10_4))
        # col 5
        triplets.append((row2, col5, weight * s2_5))
        triplets.append((row5, col5, weight * s5_5))
        triplets.append((row8, col5, weight * s8_5))
        triplets.append((row11, col5, weight * s11_5))
        # col 6
        triplets.append((row3, col6, weight * s3_6))
        triplets.append((row6, col6, weight * s6_6))
        triplets.append((row9, col6, weight * s9_6))
        triplets.append((row12, col6, weight * s12_6))
        # col 7
        triplets.append((row1, col7, weight * s1_7))
        triplets.append((row4, col7, weight * s4_7))
        triplets.append((row7, col7, weight * s7_7))
        triplets.append((row10, col7, weight * s10_7))
        # col 8
        triplets.append((row2, col8, weight * s2_8))
        triplets.append((row5, col8, weight * s5_8))
        triplets.append((row8, col8, weight * s8_8))
        triplets.append((row11, col8, weight * s11_8))
        # col 9
        triplets.append((row3, col9, weight * s3_9))
        triplets.append((row6, col9, weight * s6_9))
        triplets.append((row9, col9, weight * s9_9))
        triplets.append((row12, col9, weight * s12_9))
        # col 10
        triplets.append((row1, col10, weight * s1_10))
        triplets.append((row4, col10, weight * s4_10))
        triplets.append((row7, col10, weight * s7_10))
        triplets.append((row10, col10, weight * s10_10))
        # col 11
        triplets.append((row2, col11, weight * s2_11))
        triplets.append((row5, col11, weight * s5_11))
        triplets.append((row8, col11, weight * s8_11))
        triplets.append((row11, col11, weight * s11_11))
        # col 12
        triplets.append((row3, col12, weight * s3_12))
        triplets.append((row6, col12, weight * s6_12))
        triplets.append((row9, col12, weight * s9_12))
        triplets.append((row12, col12, weight * s12_12))

        # def triplet(i, j, val):
        #     return (i, j, weight * val)
        #
        # # Column layout mirrors C++: each variable group (x/y/z for each vertex)
        # # 12 rows × 12 cols with sparsity
        #
        # rows = [
        #     vi, vi + 1, vi + 2,
        #     vj, vj + 1, vj + 2,
        #     vk, vk + 1, vk + 2,
        #     vl, vl + 1, vl + 2,
        # ]
        # cols = rows  # symmetric block layout
        #
        # grads = [
        #     (d11, d12, d13),
        #     (d21, d22, d23),
        #     (d31, d32, d33),
        #     (_d11_d21_d31, _d12_d22_d32, _d13_d23_d33),
        # ]
        #
        # triplets = []
        # for i in range(4):
        #     gi = grads[i]
        #     for j in range(4):
        #         gj = grads[j]
        #         for a in range(3):  # x/y/z of i
        #             for b in range(3):  # x/y/z of j
        #                 row = rows[3 * i + a]
        #                 col = cols[3 * j + b]
        #                 val = gi[a] * gj[b]
        #                 triplets.append(triplet(row, col, val))

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
