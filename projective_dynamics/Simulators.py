# pd/solver.py

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import lil_matrix
from utils import check_dir_exists
import os

def flatten(p: np.ndarray) -> np.ndarray:
    """
    Converts an (N, 3) matrix to a (3N,) vector by stacking rows.
    """
    return p.reshape(-1)

def unflatten(q: np.ndarray) -> np.ndarray:
    """
    Converts a (3N,) vector back to an (N, 3) matrix.
    """
    return q.reshape(-1, 3)

class Solver:
    def __init__(self):
        self.model = None
        self.dirty = True
        self.A = None
        self.cholesky = None
        self.dt = None
        self.frame = 0

    def set_model(self, model):
        self.model = model
        self.set_dirty()

    def set_dirty(self):
        self.dirty = True

    def set_clean(self):
        self.dirty = False

    def ready(self):
        return not self.dirty

    def prepare(self, args, store_fom_info=False, record_path=None):

        def store_assembly_matrices():
            """store a .npz contains assembly matrices for all used constraint types"""
            assert record_path is not None
            check_dir_exists(record_path)

            matrices = {}
            file_name = "assembly_ST"
            if self.model.has_verts_bending_constraints :
                matrices["verts_bending" ] = self.model.verts_bending_assembly_ST
                # file_name = file_name + "verts_bending_"

            if self.model.has_edge_spring_constraints :
                matrices["edge_spring" ] = self.model.edge_spring_assembly_ST
                # file_name = file_name + "edge_spring_"

            if self.model.has_tris_strain_constraints :
                matrices["tris_strain" ] = self.model.tris_strain_assembly_ST
                # file_name = file_name + "tris_strain_"

            if self.model.has_tets_strain_constraints:
                matrices["tets_strain"] = self.model.tets_strain_assembly_ST
                # file_name = file_name + "tets_strain_"

            if self.model.has_tets_deformation_gradient_constraints :
                matrices["tets_deformation_gradient" ] = self.model.tets_deformation_gradient_assembly_ST
                # file_name = file_name + "tets_deformation_gradient_"

            np.savez(os.path.join(record_path , file_name+".npz") , **matrices)

        if store_fom_info:
            store_assembly_matrices()

        self.dt = args.dt

        mass = self.model.mass
        N = self.model.positions.shape[0]

        dt2_inv = 1.0 / (self.dt * self.dt)
        A_triplets = []

        for constraint in self.model.constraints:
            A_triplets += constraint.get_wi_SiT_AiT_Ai_Si()

        for i in range(N):
            A_triplets.append((3 * i + 0, 3 * i + 0, mass[i] * dt2_inv))
            A_triplets.append((3 * i + 1, 3 * i + 1, mass[i] * dt2_inv))
            A_triplets.append((3 * i + 2, 3 * i + 2, mass[i] * dt2_inv))

        rows, cols, data = zip(*A_triplets)
        A = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(3 * N, 3 * N))

        self.cholesky = scipy.sparse.linalg.factorized(A)

        self.set_clean()

    def step(self, fext, num_iterations=10, store_stacked_projections=False, record_path=None):
        velocities = self.model.velocities
        mass = self.model.mass
        constraints = self.model.constraints
        N = self.model.positions.shape[0]
        self.model.positions_corrections = np.zeros_like(self.model.positions)

        dt = self.dt
        dt_inv = 1.0 / dt
        dt2 = dt * dt
        dt2_inv = 1.0 / dt2

        a = fext / mass[:, None]  # elementwise divide
        explicit = self.model.positions + dt * velocities + dt2 * a

        for v in range(self.model.positions.shape[0]):
            self.model.resolve_collision(v, explicit, self.model.positions_corrections)

        sn = flatten(explicit)

        masses = np.zeros(3 * N)
        for i in range(N):
            masses[3 * i:3 * i + 3] = dt2_inv * mass[i] * sn[3 * i:3 * i + 3]

        q = sn.copy()

        if store_stacked_projections:
            if self.model.has_verts_bending_constraints:
                assert self.model.verts_bending_assembly_ST is not None
                self.model.verts_bending_stacked_p = np.zeros((self.model.verts_bending_assembly_ST.shape[1], 3))

                for i, c in enumerate(self.model.verts_bending_constraints):
                    self.model.verts_bending_stacked_p[i,:] = c.get_pi(q)

                np.savez(os.path.join(record_path ,"verts_bending_p_"+str(self.frame)+".npz") , self.model.verts_bending_stacked_p)

            if self.model.has_edge_spring_constraints :
                assert self.model.edge_spring_assembly_ST is not None
                self.model.edge_spring_stacked_p = np.zeros((self.model.edge_spring_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.edge_spring_constraints):
                    self.model.edge_spring_stacked_p[i, :] = c.get_pi(q)

                np.savez(os.path.join(record_path ,"edge_spring_p_"+str(self.frame)+".npz") , self.model.edge_spring_stacked_p)

            if self.model.has_tris_strain_constraints :
                assert self.model.tris_strain_assembly_ST is not None
                self.model.tris_strain_stacked_p = np.zeros((self.model.tris_strain_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.tris_strain_constraints):
                    self.model.tris_strain_stacked_p[i, :] = c.get_pi(q)

                np.savez(os.path.join(record_path ,"tris_strain_p_"+str(self.frame)+".npz") , self.model.tris_strain_stacked_p)

            if self.model.has_tets_strain_constraints:
                assert self.model.tets_strain_assembly_ST is not None
                self.model.tets_strain_stacked_p = np.zeros((self.model.tets_strain_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.tets_strain_constraints):
                    self.model.tets_strain_stacked_p[i, :] = c.get_pi(q)

                np.savez(os.path.join(record_path ,"tets_strain_p_"+str(self.frame)+".npz") , self.model.tets_strain_stacked_p)

            if self.model.has_tets_deformation_gradient_constraints :
                assert self.model.tets_deformation_gradient_assembly_ST is not None
                self.model.tets_deformation_gradient_stacked_p = np.zeros((self.model.tets_deformation_gradient_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.tets_deformation_gradient_constraints):
                    self.model.tets_deformation_gradient_stacked_p[i, :] = c.get_pi(q)

                np.savez(os.path.join(record_path ,"tet_deformation_gradient_p_"+str(self.frame)+".npz") , self.model.tets_deformation_gradient_stacked_p)



        for _ in range(num_iterations):
            #
            # # Triplets version:
            # b = np.zeros(3 * N)
            # for constraint in constraints:
            #     constraint.project_wi_SiT_AiT_Bi_pi(q, b)
            # b += masses

            #3d
            b = np.zeros((N, 3))
            for constraint in constraints:
                constraint.project_wi_SiT_pi(q, b)
            b += unflatten(masses)

            q = self.cholesky(b.flatten())

        q_next = unflatten(q)
        self.model.velocities = (q_next - self.model.positions) * dt_inv
        self.model.positions = q_next
        print(self.frame)
        self.frame +=1
