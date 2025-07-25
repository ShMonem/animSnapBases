# pd/solver.py

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import lil_matrix
from scipy.linalg import lu_factor, lu_solve
from utils import check_dir_exists
import os
from joblib import Parallel, delayed
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

class animSnapBasesSolver:
    def __init__(self):
        self.model = None
        self.dirty = True
        self.A = None
        self.cholesky = None
        self.dt = None
        self.frame = 0

        self.reduced_position = False
        self.U = None # animSnap positions basis U
        self.num_pos_basis_modes = -1
        self.V = None # animSnap constraints projection basis V
        self.num_constrproj_basis_blocks = -1

        # which constraint projections are reduced
        self.reduced_local_step = False
        self.reduced_verts_bending = False
        self.projecting_mat_verts_bending = None    # can be either "U^T S^T V" or  "S^T V"
        self.mapped_indices_verts_bending = None    # when mesh is not closed, not all verts are constrained
        self.interpolation_alpha_verts_bending = None  # original indices
        self.cholesky_list_verts_bending  = []

        self.reduced_edge_spring = False
        self.reduced_tris_strain = False
        self.reduced_tets_strain = False
        self.reduced_tets_deformation_gradient = False


    def set_model(self, model):
        self.model = model
        self.set_dirty()

    def set_dirty(self):
        self.dirty = True

    def set_clean(self):
        self.dirty = False

    def ready(self):
        return not self.dirty

    def prepare_full_global(self, args):

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
        return scipy.sparse.csc_matrix((data, (rows, cols)), shape=(3 * N, 3 * N))   # (mass/dt^2)+ Sum_i wi SiT Si

    def prepare(self, args, store_fom_info=False, record_path=None):

        full_global_mat = self.prepare_full_global(args)

        if not self.reduced_position :
            self.cholesky = scipy.sparse.linalg.factorized(full_global_mat)
        else:
            raise ValueError("animSnapBases position reduction not yet implemented")
            # read components/basis from file
            # global_data = np.load(basis_file)
            #

            #
            # print("something!")
            #
            # self.cholesky = scipy.linalg.factorized(Ut full_global_mat U)

        if not self.reduced_local_step:
            pass
        else:
            # TODO: one file has all basis types all mulity files?

            if self.model.has_verts_bending_constraints and self.reduced_verts_bending:

                local_data = np.load("constraint_projections_basis_file")
                Vj = local_data["components"]  # shape shold be (ep, mp, 3) # TODO: check
                self.interpolation_alpha_verts_bending = local_data["interpol_alphas"]
                print(f"Basis file loaded with: \n Basis shape {Vj.shape} and {self.interpolation_alpha_verts_bending} interpolation points.")

                """  
                in case of "vertbend" for non-closed meshes,
                mapping between index and its order in the list of constrained_elements is required.
                when we call constraints to stack them, we use alphas,
                while when we compute V[Pt,:] now Pt/mapped_indices takes alpha to where it appears in the V rows
                """
                self.mapped_indices_verts_bending = [i for i, val in enumerate(self.model.verts_bending_indicies) if
                                       val in self.interpolation_alpha_verts_bending]

                if not self.reduced_global_step:
                    # ST (N, ep) @ V (ep, mp, 3) --> (N, mp, 3) # TODO : check (N, m.p, 3)
                    self.projecting_mat_verts_bending = np.einsum('ne,emi->nmi', self.model.verts_bending_assembly_ST, Vj)
                else:
                    # U^T (r, N, 3) @ S^T (N, ep) --> (r, ep, 3)
                    UtSt = np.einsum('rni,ne->rei', self.U.T, self.model.verts_bending_assembly_ST)
                    # U^T S^T(r, ep, 3) @ Vj (ep, mp, 3) --> (r, mp, 3)
                    self.projecting_mat_verts_bending =  np.einsum('rei,emi->rmi', UtSt, Vj) # TODO : check (r, m.p, 3)

                PtV = Vj[:, self.mapped_indices_verts_bending , :]   # TODO : check (m.p, m.p, 3)

                for d in range(3):
                    self.cholesky_list_verts_bending.append(lu_factor(PtV))

                print("Solver list created", self.cholesky_list_verts_bending)

            if self.model.has_edge_spring_constraints and self.reduced_edge_spring:
                raise ValueError("not yet implemented! spring")
            if self.model.has_tris_strain_constraints and self.reduced_tris_strain:
                raise ValueError("not yet implemented! strain")
            if self.model.has_tets_strain_constraints and self.reduced_tets_strain:
                raise ValueError("not yet implemented! strain")
            if self.model.has_tets_deformation_gradient_constraints and self.reduced_tets_deformation_gradient:
                raise ValueError("not yet implemented! gradient")

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

        def get_group_ST_p(q_t, group_constraints, constraint_dim, ST):
            """
            Args:
                group_constraints:
                constraint_dim:
                ST:
                q_t:
                rhs:

            Returns:
                ST p: full "non-reduced" constraint projection computation for one constraint group
            """
            p = np.zeros((ST.shape[1], 3))
            for i, c in enumerate(group_constraints):
                p[constraint_dim * i:constraint_dim * i + constraint_dim, :]  = c.get_pi(q_t)
            if store_stacked_projections:
                np.savez(os.path.join(record_path, "positional_p_" + str(self.frame) + ".npz"),p)
            # update constraints projection term
            return ST @ p

        def get_group_reduced_term(q_t, group_constraints, constraint_dim, constrained_alphas, projection_mat, solver_list):
            """
            Args:
                group_constraints:
                constraint_dim:
                constrained_alphas:
                projection_mat:
                solver_list:
                q_t:
                rhs:

            Returns:
                U^T S^T V p_tilde: if position reduction is used
                S^T V p_tilde: otherwise

            """
            p = np.zeros((len(constrained_alphas) * constraint_dim , 3))  # (m.p, 3)

            for i, alpha in enumerate(constrained_alphas):
                c = group_constraints[alpha]
                p[constraint_dim * i:constraint_dim * i + constraint_dim, :] = c.get_pi(q_t)

            def compute_rhs_column(d):
                return projection_mat[:, :, d] @ lu_solve(solver_list[d], p[:, d])

            rhs_cols = Parallel(n_jobs=3)(delayed(compute_rhs_column)(d) for d in range(3))
            temp = np.column_stack(rhs_cols)
            return temp


        def get_constraints_projection_term(q_t, rhs):
            if self.model.has_positional_constraints:
                assert self.model.positional_assembly_ST is not None
                self.model.positional_stacked_p = np.zeros((self.model.positional_assembly_ST.shape[1], 3))

                for i, c in enumerate(self.model.positional_constraints):
                    self.model.positional_stacked_p[i, :] = c.get_pi(q_t)
                if store_stacked_projections:
                    np.savez(os.path.join(record_path, "positional_p_" + str(self.frame) + ".npz"),
                             self.model.positional_stacked_p)
                # update constraints projection term
                rhs += self.model.positional_assembly_ST @ self.model.positional_stacked_p

            if self.model.has_verts_bending_constraints:
                assert self.model.verts_bending_assembly_ST is not None

                if not self.reduced_verts_bending:
                    rhs += get_group_ST_p(q_t, self.model.verts_bending_constraints, 1, self.model.verts_bending_assembly_ST)
                else:
                    rhs += get_group_reduced_term(q_t, self.model.verts_bending_constraints, 1,
                                                  self.interpolation_alpha_verts_bending,
                                                  self.projecting_mat_verts_bending, self.cholesky_list_verts_bending)

            if self.model.has_edge_spring_constraints:
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

            if self.model.has_tris_strain_constraints:
                assert self.model.tris_strain_assembly_ST is not None
                self.model.tris_strain_stacked_p = np.zeros((self.model.tris_strain_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.tris_strain_constraints):
                    self.model.tris_strain_stacked_p[2 * i:2 * i + 2, :] = c.get_pi(q_t)
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
                    self.model.tets_strain_stacked_p[3 * i:3 * i + 3, :] = c.get_pi(q_t)
                if store_stacked_projections:
                    tets_strain_p[str(self.frame)] = self.model.tets_strain_stacked_p
                    if self.frame == max_p_snapshots_num:
                        np.savez(os.path.join(record_path, "tets_strain_p" + ".npz"), **tets_strain_p)

                    # np.savez(os.path.join(record_path ,"tets_strain_p_"+str(self.frame)+".npz") , self.model.tets_strain_stacked_p)
                # update constraints projection term
                rhs += self.model.tets_strain_assembly_ST @ self.model.tets_strain_stacked_p

            if self.model.has_tets_deformation_gradient_constraints:
                assert self.model.tets_deformation_gradient_assembly_ST is not None
                self.model.tets_deformation_gradient_stacked_p = np.zeros(
                    (self.model.tets_deformation_gradient_assembly_ST.shape[1], 3))
                for i, c in enumerate(self.model.tets_deformation_gradient_constraints):
                    self.model.tets_deformation_gradient_stacked_p[3 * i:3 * i + 3, :] = c.get_pi(q_t)
                if store_stacked_projections:
                    tets_deformation_gradient_p[str(self.frame)] = self.model.tets_deformation_gradient_stacked_p
                    if self.frame == max_p_snapshots_num:
                        np.savez(os.path.join(record_path, "tets_deformation_gradient_p" + ".npz"),
                                 **tets_deformation_gradient_p)

                    # np.savez(os.path.join(record_path ,"tet_deformation_gradient_p_"+str(self.frame)+".npz") , self.model.tets_deformation_gradient_stacked_p)
                # update constraints projection term
                rhs += self.model.tets_deformation_gradient_assembly_ST @ self.model.tets_deformation_gradient_stacked_p

        for _ in range(num_iterations):
            b = np.zeros((N, 3))

            if use_3d_rhs_form:
                get_constraints_projection_term(q, b)  # much faster and more efficient
            else:
                for constraint in constraints:
                    constraint.project_wi_SiT_pi(q, b)
            b += unflatten(masses)

            q = self.cholesky(b.flatten())

        q_next = unflatten(q)
        q_next = self.model.resolve_self_collision_fast(q_next)
        q_next = self.model.resolve_triangle_self_collisions(q_next)
        self.model.velocities = (q_next - self.model.positions) * dt_inv
        self.model.positions = q_next
        print(self.frame)
        self.frame += 1


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
