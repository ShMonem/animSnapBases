# pd/solver.py

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import lil_matrix
from utils import check_dir_exists
import os

max_p_snapshots_num = 220
verts_bending_p = {}
edge_spring_p = {}
tris_strain_p = {}
tets_strain_p = {}
tets_deformation_gradient_p = {}
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
            # if self.model.has_positional_constraints :
            #     matrices["positional" ] = self.model.positional_assembly_ST

            if self.model.has_verts_bending_constraints :
                matrices["verts_bending" ] = self.model.verts_bending_assembly_ST
                np.savez(os.path.join(record_path , "verts_bending_constrained_indices.npz"), indices=self.model.verts_bending_indicies)

            if self.model.has_edge_spring_constraints :
                matrices["edge_spring" ] = self.model.edge_spring_assembly_ST

            if self.model.has_tris_strain_constraints :
                matrices["tris_strain" ] = self.model.tris_strain_assembly_ST

            if self.model.has_tets_strain_constraints:
                matrices["tets_strain"] = self.model.tets_strain_assembly_ST

            if self.model.has_tets_deformation_gradient_constraints :
                matrices["tets_deformation_gradient" ] = self.model.tets_deformation_gradient_assembly_ST

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


    def step(self, fext, num_iterations=10, use_3d_rhs_form=True, store_stacked_projections=False, record_path=None):
        global max_p_snapshots_num, verts_bending_p, edge_spring_p, tris_strain_p, tets_strain_p, tet_deformation_gradient_p

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

        def get_sum_ST_p(q_t, rhs):
            if self.model.has_positional_constraints:
                assert self.model.positional_assembly_ST is not None
                self.model.positional_stacked_p = np.zeros((self.model.positional_assembly_ST.shape[1], 3))

                for i, c in enumerate(self.model.positional_constraints):
                    self.model.positional_stacked_p[i,:] = c.get_pi(q_t)
                if store_stacked_projections:
                    np.savez(os.path.join(record_path ,"positional_p_"+str(self.frame)+".npz") , self.model.positional_stacked_p)
                # update constraints projection term
                rhs += self.model.positional_assembly_ST @ self.model.positional_stacked_p

            if self.model.has_verts_bending_constraints:
                assert self.model.verts_bending_assembly_ST is not None
                self.model.verts_bending_stacked_p = np.zeros((self.model.verts_bending_assembly_ST.shape[1], 3))

                for i, c in enumerate(self.model.verts_bending_constraints):
                    self.model.verts_bending_stacked_p[i,:] = c.get_pi(q_t)
                if store_stacked_projections:
                    verts_bending_p[str(self.frame)] = self.model.verts_bending_stacked_p
                    if self.frame == max_p_snapshots_num:
                        np.savez(os.path.join(record_path, "verts_bending_p" + ".npz"), **verts_bending_p)

                    #np.savez(os.path.join(record_path ,"verts_bending_p_"+str(self.frame)+".npz") , self.model.verts_bending_stacked_p)
                # update constraints projection term
                rhs += self.model.verts_bending_assembly_ST @ self.model.verts_bending_stacked_p

            if self.model.has_edge_spring_constraints :
                assert self.model.edge_spring_assembly_ST is not None
                self.model.edge_spring_stacked_p = np.zeros((self.model.edge_spring_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.edge_spring_constraints):
                    self.model.edge_spring_stacked_p[i, :] = c.get_pi(q_t)
                if store_stacked_projections:
                    edge_spring_p[str(self.frame)] = self.model.edge_spring_stacked_p
                    if self.frame == max_p_snapshots_num:
                        np.savez(os.path.join(record_path, "edge_spring_p" + ".npz"), **edge_spring_p)

                    # np.savez(os.path.join(record_path ,"edge_spring_p_"+str(self.frame)+".npz") , self.model.edge_spring_stacked_p)
                # update constraints projection term
                rhs += self.model.edge_spring_assembly_ST @ self.model.edge_spring_stacked_p

            if self.model.has_tris_strain_constraints :
                assert self.model.tris_strain_assembly_ST is not None
                self.model.tris_strain_stacked_p = np.zeros((self.model.tris_strain_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.tris_strain_constraints):
                    self.model.tris_strain_stacked_p[2*i:2*i+2, :] = c.get_pi(q_t)
                if store_stacked_projections:
                    tris_strain_p[str(self.frame)] = self.model.tris_strain_stacked_p
                    if self.frame == max_p_snapshots_num:
                        np.savez(os.path.join(record_path, "tris_strain_p" + ".npz"), **tris_strain_p)

                    # np.savez(os.path.join(record_path ,"tris_strain_p_"+str(self.frame)+".npz") , self.model.tris_strain_stacked_p)
                # update constraints projection term
                rhs += self.model.tris_strain_assembly_ST @ self.model.tris_strain_stacked_p

            if self.model.has_tets_strain_constraints:
                assert self.model.tets_strain_assembly_ST is not None
                self.model.tets_strain_stacked_p = np.zeros((self.model.tets_strain_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.tets_strain_constraints):
                    self.model.tets_strain_stacked_p[3*i:3*i+3, :] = c.get_pi(q_t)
                if store_stacked_projections:
                    tets_strain_p[str(self.frame)] = self.model.tets_strain_stacked_p
                    if self.frame == max_p_snapshots_num:
                        np.savez(os.path.join(record_path, "tets_strain_p" + ".npz"), **tets_strain_p)

                    # np.savez(os.path.join(record_path ,"tets_strain_p_"+str(self.frame)+".npz") , self.model.tets_strain_stacked_p)
                # update constraints projection term
                rhs += self.model.tets_strain_assembly_ST @ self.model.tets_strain_stacked_p

            if self.model.has_tets_deformation_gradient_constraints :
                assert self.model.tets_deformation_gradient_assembly_ST is not None
                self.model.tets_deformation_gradient_stacked_p = np.zeros((self.model.tets_deformation_gradient_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.tets_deformation_gradient_constraints):
                    self.model.tets_deformation_gradient_stacked_p[3*i:3*i+3, :] = c.get_pi(q_t)
                if store_stacked_projections:
                    tets_deformation_gradient_p[str(self.frame)] = self.model.tets_deformation_gradient_stacked_p
                    if self.frame == max_p_snapshots_num:
                        np.savez(os.path.join(record_path, "tets_deformation_gradient_p" + ".npz"), **tets_deformation_gradient_p)

                    # np.savez(os.path.join(record_path ,"tet_deformation_gradient_p_"+str(self.frame)+".npz") , self.model.tets_deformation_gradient_stacked_p)
                # update constraints projection term
                rhs += self.model.tets_deformation_gradient_assembly_ST @ self.model.tets_deformation_gradient_stacked_p

        for _ in range(num_iterations):
            b = np.zeros((N, 3))

            if use_3d_rhs_form:
                get_sum_ST_p(q, b)  # much faster and more efficient
            else:
                for constraint in constraints:
                    constraint.project_wi_SiT_pi(q, b)
            b += unflatten(masses)

            q = self.cholesky(b.flatten())

        q_next = unflatten(q)
        q_next = self.model.resolve_self_collision_fast(q_next)
        q_next= self.model.resolve_triangle_self_collisions(q_next)
        self.model.velocities = (q_next - self.model.positions) * dt_inv
        self.model.positions = q_next
        print(self.frame)
        self.frame +=1
