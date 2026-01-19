# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags, issparse
from scipy.linalg import lu_factor, lu_solve, eigh, solve_triangular, solve
from utils import check_dir_exists, compute_surface_geodesics, visualize_samples
from lbs import ConstraintsProjectionSubspace, PositionsSubspace
import os
from joblib import Parallel, delayed
# from scipy.spatial import KDTree
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

verts_bending_p = {}
edge_spring_p = {}
tris_strain_p = {}
tets_strain_p = {}
tets_deformation_gradient_p = {}
positions = {}




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


class constraintProjection:
    def __init__(self, row_dim, reduced, num_components, num_samples):
        self.row_dim = row_dim   # p in (px3)
        self.is_reduced = reduced   # is constraint type reduced
        self.num_components = num_components
        self.num_samples = num_samples
        self.projection_matrix = None  # can be either "U^T S^T V" or  "S^T V"
        self.Pt = []
        self.solver_list = []
        self.mapped_indices_Pt = None #  when mesh is not closed, not all verts are constrained
        self.interpolation_alpha = None
        self.sampled_constraints = None

        # allows for single constraint type update, if you add/change constraint from a callback reset self.subspace_projection_ready to false
        self.subspace_projection_ready = False

        
class animSnapBasesSolver:
    def __init__(self, args):
        self.model = None
        self.dirty = True
        self.A = None
        self.cholesky = None
        self.animSnap_cholesky_list = []
        self.project_to_subspace_solver = []
        self.dt = None
        self.frame = 0
        self.reference_frame = 0  # to reset the count when switching between different call back experiments
        self.args = args
        self.recording_path = None
        self.record_path_has_changed = True

        # Positions reduction
        self.U = None # animSnap positions basis U
        self.has_reduced_position = args.positions_reduced
        if self.has_reduced_position:
            self.position_reduction_type = args.position_basis_type
            self.position_basis_num_components = args.num_position_components
        else:
            self.position_reduction_type = ""
            self.position_basis_num_components = -1

        # Constraint projections
        self.bending = constraintProjection(1, args.vert_bending_reduced, args.vert_bending_num_components, args.vert_bending_num_samples)
        self.spring = constraintProjection(1, args.edge_spring_reduced, args.edge_spring_num_components, args.edge_spring_num_samples)
        self.tris_strain = constraintProjection(2, args.tri_strain_reduced, args.tri_strain_num_components, args.tri_strain_num_samples)
        self.tets_strain = constraintProjection(3, args.tet_strain_reduced, args.tet_strain_num_components, args.tet_strain_num_samples)
        self.tets_deformation_gradient = constraintProjection(3, args.tet_deformation_reduced, args.tet_deformation_num_components, args.tet_deformation_num_samples)

        self.has_reduced_constraint_projections = any([self.bending.is_reduced ,
                                        self.spring.is_reduced,
                                        self.tris_strain.is_reduced ,
                                        self.tets_strain.is_reduced,
                                        self.tets_deformation_gradient.is_reduced
        ])
        self.constraint_projection_reduction_type = args.constraint_projection_basis_type

        self.constraint_subspace_ready = False
        self.position_subspace_ready =False
        # self.constraints_ready = False

        self.store_stacked_projections = False
        self.store_positions = False
        self.store_current_snapshots = False
        self.max_p_snapshots_num = args.max_p_snapshots_num
        self.max_q_snapshots_num = args.max_p_snapshots_num

        self.involved_constraints = []
        self.snapBases_interpolation_list = {"deim_pod", "deim_pod_vectorized", "deim_pca_blocks",
                                             "geom_pca_blocks_with_St", "adv_pca_blocks_with_St_partitioning", "adv_pca_blocks"}

    def set_max_recorded_frames(self, value:int):
        self.max_p_snapshots_num = value
        self.max_q_snapshots_num = value

    @staticmethod
    def reset_positions_and_constraint_projections_snapshots():
        global verts_bending_p, edge_spring_p, tris_strain_p, tets_strain_p, tets_deformation_gradient_p, positions
        verts_bending_p = {}
        edge_spring_p = {}
        tris_strain_p = {}
        tets_strain_p = {}
        tets_deformation_gradient_p = {}
        positions = {}


    def set_max_q_frames(self, value: int):
        self.max_q_snapshots_num = value

    def set_record_path(self, path: str):
        self.recording_path = path

    def set_store_p(self, value: bool):
        self.store_stacked_projections = value

    def set_store_q(self, value: bool):
        self.store_positions = value

    def set_model(self, model):
        self.model = model
        self.set_dirty()

    def set_dirty(self):
        self.dirty = True

    def set_clean(self):
        self.dirty = False

    def ready(self):
        return not self.dirty

    def set_model_constraints(self):

        # Only reset the constraint that has changed
        if self.args.vert_bending_constraint and self.model.bending_constraints_changed:
            self.model.add_vertex_bending_constraint(self.args.vert_bending_constraint_wi)
            self.model.bending_constraints_changed = False
            if self.bending.is_reduced:
                self.bending.subspace_projection_ready = False

        if self.args.edge_constraint and self.model.spring_constraints_changed:
            self.model.add_edge_spring_constrain(self.args.edge_constraint_wi)
            self.model.spring_constraints_changed = False
            if self.spring.is_reduced:
                self.spring.subspace_projection_ready = False

        if self.args.tri_strain_constraint and self.model.tris_strain_constraints_changed:
            self.model.add_tri_constrain_strain(
                self.args.sigma_min,
                self.args.sigma_max,
                self.args.strain_limit_constraint_wi)
            self.model.tris_strain_constraints_changed = False
            if self.tris_strain.is_reduced:
                self.tris_strain.subspace_projection_ready = False

        if self.args.tet_strain_constraint and self.model.tets_strain_constraints_changed:
            self.model.add_tet_constrain_strain(
                self.args.sigma_min,
                self.args.sigma_max,
                self.args.strain_limit_constraint_wi)
            self.model.tets_strain_constraints_changed = False
            if self.tets_strain.is_reduced:
                self.tets_strain.subspace_projection_ready = False

        if self.args.tet_deformation_constraint and self.model.tets_deformation_constraints_changed:
            self.model.add_tet_constrain_deformation_gradient(self.args.deformation_gradient_constraint_wi)
            self.model.tets_deformation_constraints_changed = False
            if self.tets_deformation_gradient.is_reduced:
                self.tets_deformation_gradient.subspace_projection_ready = False


        # note: positional constrained is set directly in the DeformableMesh class
        self.model.constraints = [*self.model.positional_constraints,
                                  *self.model.verts_bending_constraints,
                                  *self.model.edge_spring_constraints,
                                  *self.model.tris_strain_constraints,
                                  *self.model.tets_strain_constraints,
                                  *self.model.tets_deformation_gradient_constraints]

    """ ------------------------------------------------------------------------------
    If LBS subspace reduction is used for constraints projection
    ------------------------------------------------------------------------------ """
    def prepare_lbs_reduced_group(self, has_group_constraints, reduced_group, group_name,
                                  group_constraints, group_aux_size, assembly_ST, assembly_ST_no_weights, num_components, num_samples, specify_verts=[]):

        if has_group_constraints and reduced_group:
            group_subspace = ConstraintsProjectionSubspace(self.args.constraint_radial_r_muliplier,
                                                           self.args.constraint_basis_scale,
                                                           group_name,
                                                           self.model.positions,
                                                           self.model.faces,
                                                           self.model.elements,
                                                           num_components, num_samples)
            # compute skinning weights
            group_subspace.compute_skinning_weights()
            # compute constraint group mass matrix
            group_subspace.compute_constraint_mass_matrix(group_name, group_constraints, self.model.lumped_mass, self.model.mass_init, group_aux_size)

            # compute lbs basis "V" for the constraint group
            group_subspace.create_basis_via_skinning_weights(self.model.positions, assembly_ST_no_weights, group_constraints, group_aux_size,
                                              use_pca=True, specify_verts= specify_verts, normalization_factor= self.model.mass_normalization)

            if not self.has_reduced_position or self.position_reduction_type == "LBS":
                projecting_mat = np.einsum('ne,em->nm',assembly_ST.to_dense().cpu().detach().numpy(), group_subspace.V.to_dense().cpu().detach().numpy())
            else:
                raise ValueError(
                    "LBS reduced constraint projections can be combined only with similar position reduction.")

            # create a list of constrained elements and alpha points
            # compute interpolation solvers for lbs subspace
            group_subspace.init_constraint_group_interpolation(group_constraints, aux_size=group_aux_size)

            print(f"Created LBS interpolation basis via skinning weights for {group_name} of size {group_subspace.V.shape}.")

            return group_subspace.sampled_constraints_ids, group_subspace.sampled_constraints, projecting_mat, group_subspace.interpol_solver
        else:
            return [],[], None, []

    def prepare_lbs_reduced_verts_bending(self):
        self.bending.interpolation_alpha, self.bending.sampled_constraints, self.bending.projection_matrix, self.bending.solver_list = \
            self.prepare_lbs_reduced_group(self.model.has_verts_bending_constraints, self.bending.is_reduced,
                                       "verts_bending", self.model.verts_bending_constraints,
                                       self.bending.row_dim, self.model.verts_bending_assembly_ST, self.model.verts_bending_assembly_ST_no_weights,
                                       self.bending.num_components, self.bending.num_samples, specify_verts=self.model.verts_bending_indicies)

    def prepare_lbs_reduced_edge_spring(self):
        self.spring.interpolation_alpha, self.spring.sampled_constraints, self.spring.projection_matrix, self.spring.solver_list = \
            self.prepare_lbs_reduced_group(self.model.has_edge_spring_constraints, self.spring.is_reduced,
                                       "edge_spring", self.model.edge_spring_constraints,
                                       self.spring.row_dim, self.model.edge_spring_assembly_ST, self.model.edge_spring_assembly_ST_no_weights,
                                       self.spring.num_components, self.spring.num_samples,)

    def prepare_lbs_reduced_tris_strain(self):
        self.tris_strain.interpolation_alpha, self.tris_strain.sampled_constraints, self.tris_strain.projection_matrix, self.tris_strain.solver_list = \
            self.prepare_lbs_reduced_group(self.model.has_tris_strain_constraints, self.tris_strain.is_reduced,
                                       "tris_strain", self.model.tris_strain_constraints,
                                       self.tris_strain.row_dim, self.model.tris_strain_assembly_ST, self.model.tris_strain_assembly_ST_no_weights,
                                       self.tris_strain.num_components, self.tris_strain.num_samples)

    def prepare_lbs_reduced_tets_strain(self):
        self.tets_strain.interpolation_alpha, self.tets_strain.sampled_constraints, self.tets_strain.projection_matrix, self.tets_strain.solver_list = \
            self.prepare_lbs_reduced_group(self.model.has_tets_strain_constraints, self.tets_strain.is_reduced,
                                       "tets_strain", self.model.tets_strain_constraints,
                                       self.tets_strain.row_dim, self.model.tets_strain_assembly_ST, self.model.tets_strain_assembly_ST_no_weights,
                                       self.tets_strain.num_components, self.tets_strain.num_samples)

    def prepare_lbs_reduced_tets_deformation_gradient(self):
        (self.tets_deformation_gradient.interpolation_alpha, self.tets_deformation_gradient.sampled_constraints,
         self.tets_deformation_gradient.projection_matrix, self.tets_deformation_gradient.solver_list) = \
            self.prepare_lbs_reduced_group(self.model.has_tets_deformation_gradient_constraints, self.tets_deformation_gradient.is_reduced,
                                       "tets_deformation_gradient", self.model.tets_deformation_gradient_constraints,
                                       self.tets_deformation_gradient.row_dim,
                                       self.model.tets_deformation_gradient_assembly_ST, self.model.tets_deformation_gradient_assembly_ST_no_weights,
                                       self.tets_deformation_gradient.num_components, self.tets_deformation_gradient.num_samples)

    """ ------------------------------------------------------------------------------
     Functions for animSnap subspace reduction preparation for constraints projection
    ------------------------------------------------------------------------------ """
    def prepare_snapshots_reduced_group(self, has_group_constraints, reduced_group, is_ready, group_name, num_components,
                                        row_dim, assembly_ST, dir, file):
        """
            # generic constraints reduction preparation for all groups except verts bending
        Args:
            is_ready:
            has_group_constraints:
            reduced_group:
            group_name:
            num_components:
            row_dim:
            assembly_ST:
            dir:
            file:

        Returns:

        """
        if self.constraint_projection_reduction_type in {"deim_pod", "deim_pod_vectorized"}:
            row_dim = 1    # because these methods do not select a full block but only few row indices

        # TODO: this step can be further optimized, as "is_ready" means that only selection mat has changed
        if has_group_constraints and reduced_group and not is_ready:
            solver_list = []
            upload_file = os.path.join(dir, group_name, file)
            local_data = np.load(upload_file)
            Vj = local_data["components"].swapaxes(0, 1)[:,
                 :num_components * row_dim, :]  # shape shold be (ep, mp, 3)
            alpha_range = local_data["interpol_alpha_ranges"][num_components - 1]

            interpolation_alpha= local_data["interpol_alphas"][:alpha_range]  # for verts bending we use Pt instead of alphas

            # building Pt
            if self.constraint_projection_reduction_type in {"deim_pod", "deim_pod_vectorized"}:
                Pt = local_data["Pt"][:alpha_range]     # row indices between 0 and e.p-1
            else:
                Pt = []
                for alpha in interpolation_alpha:
                    for l in range(row_dim):
                        Pt.append(alpha * row_dim + l)

            # ST (N, ep) @ V (ep, mp, 3) --> S^T V: (N, mp, 3)
            projecting_mat = np.einsum('ne,emi->nmi',assembly_ST.to_dense().cpu().detach().numpy(), Vj)

            PtV = Vj[Pt, :, :]  # (num_interpolation_alphas > m, mp, 3)
            PtV_T = Vj[Pt, :, :].swapaxes(0, 1)  # TODO : check (m.p, m.p, 3)
            AtA = np.einsum('nai,ami->nmi', PtV_T, PtV)  # for normal equation solve

            la = 1e-8 * np.trace(AtA) / AtA.shape[0]  # scale-aware lambda (to add Tikhonov regularization)

            for d in range(3):
                # for each dim store [(lu_factor(AtA), At)]
                solver_list.append([lu_factor(AtA[:, :, d]+ la[d] * np.eye(AtA[:, :, d].shape[0])), PtV_T[:, :, d]])
            print(
                group_name+f" basis file loaded with: \n Basis shape {Vj.shape} and {interpolation_alpha.shape} interpolation points.")
            return interpolation_alpha, Pt, projecting_mat, solver_list
        else:
            return None, [], None, []

    def prepare_snapshots_reduced_verts_bending(self, dir, file):
        if self.model.has_verts_bending_constraints and self.bending.is_reduced and not self.bending.subspace_projection_ready:
            upload_file = os.path.join(dir, "verts_bending", file)
            local_data = np.load(upload_file)
            Vj = local_data["components"].swapaxes(0, 1)[:, :self.bending.num_components*self.bending.row_dim, :]   # shape shold be (ep, mp, 3)

            # self.bending.interpolation_alpha = local_data["interpol_alphas"]  # for verts bending we use Pt instead of alphas
            """  
            in case of "verts_bending" for non-closed meshes, not all verts are constrained, therefore
            mapping between index and its order in the list of constrained_elements is required.
            we compute V[Pt,:] now Pt/mapped_indices takes alpha in the StV rows to where it appears in the V rows
            """

            alpha_range = local_data["interpol_alpha_ranges"][self.bending.num_components-1]
            self.bending.mapped_indices_Pt = local_data["Pt"][:alpha_range]
            self.bending.interpolation_alpha = local_data["Pt"][:alpha_range]
            print(f"Verts bending basis file loaded with: \n Basis shape {Vj.shape} and {self.bending.mapped_indices_Pt.shape} interpolation points.")

            # ST (N, ep) @ V (ep, mp, 3) --> S^T V: (N, mp, 3)
            self.bending.projection_matrix = np.einsum('ne,emi->nmi',self.model.verts_bending_assembly_ST.toarray(), Vj)

            PtV = Vj[self.bending.mapped_indices_Pt, :, :]   # (num_interpolation_alphas > m, mp, 3)
            PtV_T = Vj[self.bending.mapped_indices_Pt, : , :].swapaxes(0,1)
            AtA = np.einsum('nai,ami->nmi',PtV_T, PtV)

            for d in range(3):
                # for each dim store [(lu_factor(AtA), At)]
                self.bending.solver_list.append([lu_factor(AtA[:, :, d]), PtV_T[:, :, d]])

    def prepare_snapshots_reduced_edge_spring(self,dir, file):
        self.spring.interpolation_alpha, self.spring.Pt, self.spring.projection_matrix, self.spring.solver_list = \
        self.prepare_snapshots_reduced_group(self.model.has_edge_spring_constraints, self.spring.is_reduced, self.spring.subspace_projection_ready,
                                   "edge_spring", self.spring.num_components, self.spring.row_dim,
                                   self.model.edge_spring_assembly_ST,  dir, file)

    def prepare_snapshots_reduced_tris_strain(self, dir, file):
        self.tris_strain.interpolation_alpha, self.tris_strain.Pt,  self.tris_strain.projection_matrix, self.tris_strain.solver_list = \
            self.prepare_snapshots_reduced_group(self.model.has_tris_strain_constraints, self.tris_strain.is_reduced, self.tris_strain.subspace_projection_ready,
                                       "tris_strain", self.tris_strain.num_components, self.tris_strain.row_dim,
                                       self.model.tris_strain_assembly_ST, dir, file)

    def prepare_snapshots_reduced_tets_strain(self, dir, file):
        self.tets_strain.interpolation_alpha, self.tets_strain.Pt, self.tets_strain.projection_matrix, self.tets_strain.solver_list = \
            self.prepare_snapshots_reduced_group(self.model.has_tets_strain_constraints, self.tets_strain.is_reduced, self.tets_strain.subspace_projection_ready,
                                       "tets_strain", self.tets_strain.num_components, self.tets_strain.row_dim,
                                       self.model.tets_strain_assembly_ST, dir, file)

    def prepare_snapshots_reduced_tets_deformation_gradient(self, dir, file):
        self.tets_deformation_gradient.interpolation_alpha, self.tets_deformation_gradient.Pt, self.tets_deformation_gradient.projection_matrix, self.tets_deformation_gradient.solver_list = \
            self.prepare_snapshots_reduced_group(self.model.has_tets_deformation_gradient_constraints, self.tets_deformation_gradient.is_reduced, self.tets_deformation_gradient.subspace_projection_ready,
                                       "tets_deformation_gradient", self.tets_deformation_gradient.num_components, self.tets_deformation_gradient.row_dim,
                                       self.model.tets_deformation_gradient_assembly_ST, dir, file)


    """ ------------------------------------------------------------------------------
    Initiations for different reduction methods for constraints projections
    ------------------------------------------------------------------------------ """
    def prepare_local_term(self, args):

        if self.constraint_projection_reduction_type in self.snapBases_interpolation_list:
            dir = args.geom_interpolation_basis_dir
            file = args.geom_interpolation_basis_file

            Parallel(n_jobs=3, backend="threading")(
                delayed(f)(dir, file) for f in [self.prepare_snapshots_reduced_verts_bending,
                                                self.prepare_snapshots_reduced_edge_spring,
                                                self.prepare_snapshots_reduced_tris_strain,
                                                self.prepare_snapshots_reduced_tets_strain,
                                                self.prepare_snapshots_reduced_tets_deformation_gradient]
            )
        elif self.constraint_projection_reduction_type in {"LBS"}:

            Parallel(n_jobs=3, backend="threading")(
                delayed(f)() for f in [self.prepare_lbs_reduced_verts_bending,
                                    self.prepare_lbs_reduced_edge_spring,
                                    self.prepare_lbs_reduced_tris_strain,
                                    self.prepare_lbs_reduced_tets_strain,
                                    self.prepare_lbs_reduced_tets_deformation_gradient]
                )
        else:
            raise ValueError("Unknown reduction type for constraint projections")


    """ ------------------------------------------------------------------------------
    Building global matrix and project to position subspace if used
    ------------------------------------------------------------------------------ """
    def prepare_global_matrix(self, args):
        """
            # One time system matrix preparation step (when masses change, the global matrix will be re-computed)
        Args:
            args:

        Returns:

        """
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
        full_global_mat = sp.csc_matrix((data, (rows, cols)), shape=(3 * N, 3 * N))   # (mass/dt^2)+ Sum_i wi SiT Si

        if not self.has_reduced_position:
            self.cholesky = sp.linalg.factorized(full_global_mat)
        else:

            if not self.position_subspace_ready:  # Basis computed/loaded only once
                if self.position_reduction_type in {"PCA"}:

                    def build_U_3N_x_3M(U):
                        # U: (N, r, 3)
                        N, r, _ = U.shape
                        U_hat = np.zeros((3 * N, r))

                        Ux = U[:, :, 0]
                        Uy = U[:, :, 1]
                        Uz = U[:, :, 2]

                        U_hat[0::3] = Ux  # rows 0,3,6,... = x
                        U_hat[1::3] = Uy  # rows 1,4,7,... = y
                        U_hat[2::3] = Uz  # rows 2,5,8,... = z

                        return U_hat

                    dir = args.geom_positions_basis_dir
                    file = args.geom_positions_basis_file
                    num_components = self.position_basis_num_components
                    upload_file = os.path.join(dir, file)
                    local_data = np.load(upload_file)
                    U_tmp = local_data["components"].swapaxes(0, 1)[:,:num_components, :]  # (N, r, 3)
                    # Include initial position as one of the basis
                    U_tmp = np.concatenate([U_tmp, self.model.init_positions[:, np.newaxis, :]  ], axis=1)
                    # re-structure basis so that we can solve once for flatten positions
                    self.U = build_U_3N_x_3M(U_tmp)
                    print(f"Created animSnap positions basis of size {U_tmp.shape}.")

                elif self.position_reduction_type == "LBS":
                    pos_subspace = PositionsSubspace(self.args.pos_radial_r_muliplier,self.model.positions, faces=self.model.faces,
                                                     tets=self.model.elements, num_samples=self.position_basis_num_components)
                    pos_subspace.create_basis_via_skinning_weights()
                    self.U = pos_subspace.U
                    print(  f"Created LBS positions basis via skinning weights of size {self.U.shape}.")

                else:
                    raise ValueError("Position reduction not yet implemented")

                self.position_subspace_ready = True

            # if full_global changed update the solver only
            UtMU = self.U.T @ full_global_mat @ self.U

            if issparse(full_global_mat):
                tr = UtMU.diagonal().sum()
            else:
                tr = np.trace(UtMU)
            la = 1e-8 * tr / UtMU.shape[0]  # scale-aware lambda (add Tikhonov regularization)

            self.cholesky = sp.linalg.factorized(UtMU + la * np.eye(UtMU.shape[0]))

    """ ------------------------------------------------------------------------------
    Building global matrix and compute/initiate subspace reduction methos for positions and/or constraints' projections
    ------------------------------------------------------------------------------ """
    def prepare(self, args, store_fom_info=False, record_path=None):

        def store_assembly_matrices():
            """store a .npz contains assembly matrices for all used constraint types"""
            if store_fom_info:
                assert record_path is not None
                check_dir_exists(record_path)

            matrices = {}
            file_name = "assembly_ST"

            if self.model.has_verts_bending_constraints :
                matrices["verts_bending" ] = self.model.verts_bending_assembly_ST.to_dense().cpu().detach().numpy()
                np.savez(os.path.join(record_path , "verts_bending_constrained_indices.npz"),
                         indices=self.model.verts_bending_indicies)

            if self.model.has_edge_spring_constraints :
                matrices["edge_spring" ] = self.model.edge_spring_assembly_ST.to_dense().cpu().detach().numpy()

            if self.model.has_tris_strain_constraints :
                matrices["tris_strain" ] = self.model.tris_strain_assembly_ST.to_dense().cpu().detach().numpy()

            if self.model.has_tets_strain_constraints:
                matrices["tets_strain"] = self.model.tets_strain_assembly_ST.to_dense().cpu().detach().numpy()

            if self.model.has_tets_deformation_gradient_constraints :
                matrices["tets_deformation_gradient" ] = self.model.tets_deformation_gradient_assembly_ST.to_dense().cpu().detach().numpy()

            np.savez(os.path.join(record_path , file_name+".npz") , **matrices)

            matrices_no_w = {}
            file_name_no_w = "assembly_ST_no_w"
            # if self.model.has_positional_constraints :
            #     matrices["positional" ] = self.model.positional_assembly_ST

            if self.model.has_verts_bending_constraints:
                matrices_no_w["verts_bending"] = self.model.verts_bending_assembly_ST_no_weights.to_dense().cpu().detach().numpy()

            if self.model.has_edge_spring_constraints:
                matrices_no_w["edge_spring"] = self.model.edge_spring_assembly_ST_no_weights.to_dense().cpu().detach().numpy()

            if self.model.has_tris_strain_constraints:
                matrices_no_w["tris_strain"] = self.model.tris_strain_assembly_ST_no_weights.to_dense().cpu().detach().numpy()

            if self.model.has_tets_strain_constraints:
                matrices_no_w["tets_strain"] = self.model.tets_strain_assembly_ST_no_weights.to_dense().cpu().detach().numpy()

            if self.model.has_tets_deformation_gradient_constraints:
                matrices_no_w[
                    "tets_deformation_gradient"] = self.model.tets_deformation_gradient_assembly_ST_no_weights.to_dense().cpu().detach().numpy()

            np.savez(os.path.join(record_path, file_name_no_w + ".npz"), **matrices_no_w)


        if store_fom_info:
            if self.record_path_has_changed:
                store_assembly_matrices()
                self.record_path_has_changed = False
            self.set_store_p(store_fom_info)

        # triggers the solver to re-prepare the global matrix and update selection matrices if required
        if self.model.constraints_changed:
            # called only once
            # Apply any desired constraints
            self.model.immobilize()
            self.model.clear_constraints()
            self.model.reset_constraints_attributes()
            # sets constraints from args
            self.set_model_constraints()
            self.model.constraints_changed = False
            self.set_dirty()

        if self.has_reduced_constraint_projections and not self.constraint_subspace_ready:
            self.prepare_local_term(args)
            self.constraint_subspace_ready = True

        if self.dirty:
            # global term computation is called every time mass matrix is changed
            self.prepare_global_matrix(args)

        self.set_clean()


    """ ------------------------------------------------------------------------------
    Full dimension projection for used constraints
    ------------------------------------------------------------------------------ """
    def get_group_ST_p(self, q_t, group_constraints, constraint_dim, ST, name, list={} ):
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
        # p = np.zeros((ST.shape[1], 3))
        # for i, c in enumerate(group_constraints):
        #     p[constraint_dim * i:constraint_dim * i + constraint_dim, :]  = c.get_pi(q_t)

        p = torch.zeros((ST.shape[1], 3), dtype=torch.float32, device=ST.device)
        for i, c in enumerate(group_constraints):
            p[constraint_dim * i:constraint_dim * i + constraint_dim, :] = torch.as_tensor(c.get_pi(q_t),
                                                                                           dtype=torch.float32,
                                                                                           device=ST.device)

        if self.store_stacked_projections:
            list[str(self.frame)] = p.to_dense().cpu().detach().numpy()
            if self.frame == self.max_p_snapshots_num or self.store_current_snapshots:
                np.savez(os.path.join(self.recording_path, name + ".npz"), **list)
                if self.frame == self.max_p_snapshots_num: self.set_store_p(False)
                print(f"Frame {self.frame} : FOM snapshots with size{len(list)}, \n... stored to directory", os.path.join(self.recording_path, name + ".npz") )
                if self.store_current_snapshots:
                    self.store_current_snapshots = False
                    self.reset_positions_and_constraint_projections_snapshots()

        # update constraints projection term
        # if ST is sparse:
        result = torch.sparse.mm(ST, p).cpu().detach().numpy()
        return result #ST @ p


    """ ------------------------------------------------------------------------------
     Reduced constraints' projections
    ------------------------------------------------------------------------------ """

    def get_group_reduced_term(self, q_t, group_constraints, constraint_dim, constrained_alphas, constrained_Pt,
                               projection_mat, solver_list, constrained_samples=None):
        """
        Args:
            constrained_samples:
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

        if self.constraint_projection_reduction_type in self.snapBases_interpolation_list:

            if self.constraint_projection_reduction_type in {"deim_pod", "deim_pod_vectorized"}:
                p = np.zeros((len(constrained_alphas) , 3))  # (m.p, 3)
                for i, alpha in enumerate(constrained_alphas):
                    c = group_constraints[alpha]
                    p[i, :] = c.get_pi(q_t)[constrained_Pt[i]%constraint_dim, :]
            else:
                p = np.zeros((len(constrained_alphas) * constraint_dim , 3))  # (m.p, 3)

                for i, alpha in enumerate(constrained_alphas):
                    c = group_constraints[alpha]
                    p[constraint_dim * i:constraint_dim * i + constraint_dim, :] = c.get_pi(q_t)

            def compute_rhs_column(d):
                return projection_mat[:, :, d] @ lu_solve(solver_list[d][0], solver_list[d][1] @p[:, d])

            rhs_cols = Parallel(n_jobs=3)(delayed(compute_rhs_column)(d) for d in range(3))
            temp = np.column_stack(rhs_cols)
            return temp
        elif self.constraint_projection_reduction_type in {"LBS"}:
            p = np.zeros((len(constrained_alphas) * constraint_dim, 3))  # (m.p, 3)

            if constrained_samples is not None:
                for i, c in enumerate(constrained_samples):
                    p[constraint_dim * i:constraint_dim * i + constraint_dim, :] = c.get_pi(q_t)
            else:
                for i, alpha in enumerate(constrained_alphas):
                    c = group_constraints[alpha]
                    p[constraint_dim * i:constraint_dim * i + constraint_dim, :] = c.get_pi(q_t)

            def compute_rhs_column(d):
                # in this case projection is only two dim mat
                return solve_triangular(solver_list[0], solver_list[1] @ p[:, d])

            rhs_cols = Parallel(n_jobs=3)(delayed(compute_rhs_column)(d) for d in range(3))
            temp = np.column_stack(rhs_cols)
            return projection_mat @ temp

        else:
            raise  ValueError("Unknown constraint projection interpolation method")


    """ ------------------------------------------------------------------------------
    Following functions directs the computations to the proper method for projections and allows parallelization
    ------------------------------------------------------------------------------ """
    def project_to_positional_constraint_manifold(self, q_t):
        if self.model.has_positional_constraints:
            assert self.model.positional_assembly_ST is not None
            self.model.positional_stacked_p = np.zeros((self.model.positional_assembly_ST.shape[1], 3))

            for i, c in enumerate(self.model.positional_constraints):
                self.model.positional_stacked_p[i, :] = c.get_pi(q_t, self.frame)

            # update constraints projection term
            return self.model.positional_assembly_ST @ self.model.positional_stacked_p
        return np.zeros_like(unflatten(q_t))

    def project_to_vertex_bending_manifold(self, q_t):

        if self.model.has_verts_bending_constraints:
            if not self.bending.is_reduced:
                return self.get_group_ST_p(q_t, self.model.verts_bending_constraints, self.bending.row_dim,
                                      self.model.verts_bending_assembly_ST, name="verts_bending_p",
                                      list=verts_bending_p)
            else:
                return self.get_group_reduced_term(q_t, self.model.verts_bending_constraints, self.bending.row_dim,
                                              self.bending.interpolation_alpha, self.bending.mapped_indices_Pt,
                                              self.bending.projection_matrix, self.bending.solver_list, self.bending.sampled_constraints)
        return np.zeros_like(unflatten(q_t))

    def project_to_edge_spring_manifold(self, q_t):
        if self.model.has_edge_spring_constraints:
            if not self.spring.is_reduced:
                return self.get_group_ST_p(q_t, self.model.edge_spring_constraints, self.spring.row_dim,
                                           self.model.edge_spring_assembly_ST, name="edge_spring_p",
                                           list=edge_spring_p)
            else:
                return self.get_group_reduced_term(q_t, self.model.edge_spring_constraints, self.spring.row_dim,
                                                   self.spring.interpolation_alpha, self.spring.Pt,
                                                   self.spring.projection_matrix,
                                                   self.spring.solver_list, self.spring.sampled_constraints)
        return np.zeros_like(unflatten(q_t))

    def project_to_triangles_strain_manifold(self, q_t):
        if self.model.has_tris_strain_constraints:
            if not self.tris_strain.is_reduced:
                return self.get_group_ST_p(q_t, self.model.tris_strain_constraints, self.tris_strain.row_dim,
                                           self.model.tris_strain_assembly_ST, name="tris_strain_p",
                                           list=tris_strain_p)
            else:
                return self.get_group_reduced_term(q_t, self.model.tris_strain_constraints, self.tris_strain.row_dim,
                                                   self.tris_strain.interpolation_alpha, self.tris_strain.Pt,
                                                   self.tris_strain.projection_matrix,
                                                   self.tris_strain.solver_list, self.tris_strain.sampled_constraints)
        return np.zeros_like(unflatten(q_t))

    def project_to_tetrahedrons_strain_manifold(self, q_t):
        if self.model.has_tets_strain_constraints:
            if not self.tets_strain.is_reduced:
                return self.get_group_ST_p(q_t, self.model.tets_strain_constraints, self.tets_strain.row_dim,
                                           self.model.tets_strain_assembly_ST, name="tets_strain_p",
                                           list=tets_strain_p)
            else:
                return self.get_group_reduced_term(q_t, self.model.tets_strain_constraints, self.tets_strain.row_dim,
                                                   self.tets_strain.interpolation_alpha, self.tets_strain.Pt,
                                                   self.tets_strain.projection_matrix,
                                                   self.tets_strain.solver_list, self.tets_strain.sampled_constraints)
        return np.zeros_like(unflatten(q_t))

    def project_to_tetrahedrons_deformation_gradient_manifold(self, q_t):
        if self.model.has_tets_deformation_gradient_constraints:
            if not self.tets_deformation_gradient.is_reduced:
                return self.get_group_ST_p(q_t, self.model.tets_deformation_gradient_constraints, self.tets_deformation_gradient.row_dim,
                                           self.model.tets_deformation_gradient_assembly_ST, name="tets_deformation_gradient_p",
                                           list=tets_deformation_gradient_p)
            else:
                return self.get_group_reduced_term(q_t, self.model.tets_deformation_gradient_constraints, self.tets_deformation_gradient.row_dim,
                                                   self.tets_deformation_gradient.interpolation_alpha, self.tets_deformation_gradient.Pt,
                                                   self.tets_deformation_gradient.projection_matrix,
                                                   self.tets_deformation_gradient.solver_list, self.tets_deformation_gradient.sampled_constraints)
        return np.zeros_like(unflatten(q_t))


    """ ------------------------------------------------------------------------------
    One Newton's solver step
    ------------------------------------------------------------------------------ """
    def step(self, fext, num_iterations=1, use_3d_rhs_form=True):

        N = self.model.positions.shape[0]
        self.model.positions_corrections = np.zeros_like(self.model.positions)

        dt = self.dt
        dt_inv = 1.0 / dt
        dt2 = dt * dt
        dt2_inv = 1.0 / dt2

        a = fext / self.model.mass[:, None]  # acceleration
        explicit = self.model.positions + dt * self.model.velocities + dt2 * a

        for v in range(self.model.positions.shape[0]):
                self.model.resolve_collision(v, explicit, self.model.positions_corrections)

        sn = flatten(explicit.copy())
        rhs = np.zeros(3 * N)
        masses = np.zeros(3 * N)
        for i in range(N):
            rhs[3 * i:3 * i + 3] = dt2_inv * self.model.mass[i] * sn[3 * i:3 * i + 3]
            masses[3 * i:3 * i + 3] = self.model.mass[i] * np.ones(3)

        q = sn.copy()

        for _ in range(num_iterations):
            b = np.zeros((N, 3))

            if use_3d_rhs_form:
                # get constraints projections terms for different constraints in parallel
                result = Parallel(n_jobs=6, backend="threading")(
                    delayed(f)(q) for f in [self.project_to_positional_constraint_manifold,
                                              self.project_to_vertex_bending_manifold,
                                              self.project_to_edge_spring_manifold,
                                              self.project_to_triangles_strain_manifold,
                                              self.project_to_tetrahedrons_strain_manifold,
                                              self.project_to_tetrahedrons_deformation_gradient_manifold])
                b += sum(result)
            else:  # can be used for full sim only
                for constraint in self.model.constraints:
                    constraint.project_wi_SiT_pi(q, b)

            b += unflatten(rhs)   # M s/h² + S.T p/S.T V p   (N, 3)

            if self.has_reduced_position:
                if self.position_reduction_type in {"LBS", "PCA"}:
                    # from full to reduced
                    b = self.U.T @ b.reshape(-1)   # U.T(M s/h² + S.T p/S.T V p)   (N, 3)
                    q = self.U @ self.cholesky(b.flatten())
                else:
                    raise ValueError("unknown position reductoin")
            else:
                b = b.flatten()  # (3N,)
                q = self.cholesky(b)  # (3N,)

        q_next = unflatten(q)
        q_next = self.model.resolve_self_collision_fast(q_next)
        q_next = self.model.resolve_triangle_self_collisions(q_next)
        self.model.velocities = (q_next - self.model.positions) * dt_inv
        self.model.positions = q_next

        if self.store_positions:
            positions[str(self.frame)] = q_next.copy()
            if self.frame == self.max_q_snapshots_num or self.store_current_snapshots:
                np.savez(os.path.join(self.recording_path, "positions.npz"), **positions)
                if self.frame == self.max_q_snapshots_num: self.set_store_q(False)
                print(f"Frame {self.frame} : FOM snapshots with size{len(positions)} , \n... stored to directory", os.path.join(self.recording_path, "positions.npz") )
                if self.store_current_snapshots:
                    self.store_current_snapshots = False
                    self.reset_positions_and_constraint_projections_snapshots() # reset snapshots dict

        print(self.frame)
        self.frame += 1
