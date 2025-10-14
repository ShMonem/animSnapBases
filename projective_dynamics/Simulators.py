import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags, issparse
from scipy.linalg import lu_factor, lu_solve, eigh, solve_triangular
from utils import check_dir_exists, compute_surface_geodesics, visualize_samples
from lbs import ConstraintsProjectionSubspace, PositionsSubspace
import os
from joblib import Parallel, delayed
from scipy.spatial import KDTree

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


class constraintProjection:
    def __init__(self, row_dim, reduced, num_components):
        self.row_dim = row_dim   # p in (px3)
        self.is_reduced = reduced   # is constraint type reduced
        self.num_components = num_components
        self.projection_matrix = None  # can be either "U^T S^T V" or  "S^T V"
        self.Pt = []
        self.solver_list = []
        self.mapped_indices_Pt = None #  when mesh is not closed, not all verts are constrained
        self.interpolation_alpha = None

        
class animSnapBasesSolver:
    def __init__(self, args):
        self.model = None
        self.dirty = True
        self.A = None
        self.cholesky = None
        self.dt = None
        self.frame = 0

        # Positions reduction
        self.U = None # animSnap positions basis U
        self.reduced_position = args.positions_reduced
        if self.reduced_position:
            self.position_reduction_type = args.position_basis_type
            self.position_basis_num_components = args.num_position_components
        else:
            self.position_reduction_type = ""
            self.position_basis_num_components = -1

        # Constraint projections
        self.bending = constraintProjection(1, args.vert_bending_reduced, args.vert_bending_num_components)
        self.spring = constraintProjection(1, args.edge_spring_reduced, args.edge_spring_num_components)
        self.tris_strain = constraintProjection(2, args.tri_strain_reduced, args.tri_strain_num_components)
        self.tets_strain = constraintProjection(3, args.tet_strain_reduced, args.tet_strain_num_components)
        self.tets_deformation_gradient = constraintProjection(3, args.tet_deformation_reduced, args.tet_deformation_num_components)

        self.has_reduced_constraint_projections = any([self.bending.is_reduced ,
                                        self.spring.is_reduced,
                                        self.tris_strain.is_reduced ,
                                        self.tets_strain.is_reduced,
                                        self.tets_deformation_gradient.is_reduced
        ])
        self.constraint_projection_reduction_type = args.constraint_projection_basis_type

        self.constraint_projection_ready = False

        self.store_stacked_projections = False
        self.record_path = ""
        self.max_p_snapshots_num = args.max_p_snapshots_num

        self.snapBases_interpolation_list = {"deim_pod", "deim_pod_vectorized", "deim_pca_blocks",
                                             "geom_pca_blocks_with_St", "adv_pca_blocks_with_St_partitioning", "adv_pca_blocks"}

    def set_max_p_frames(self, value:int):
        self.max_p_snapshots_num = value

    def set_record_path(self, path: str):
        self.record_path = path
    def set_store_p(self, value: bool):
        self.store_stacked_projections = value

    def set_model(self, model):
        self.model = model
        self.set_dirty()

    def set_dirty(self):
        self.dirty = True

    def set_clean(self):
        self.dirty = False

    def ready(self):
        return not self.dirty

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

        if not self.reduced_position:
            self.cholesky = sp.linalg.factorized(full_global_mat)
        else:

            if self.position_reduction_type not in {"LBS"}:
                raise ValueError("Position reduction not yet implemented")
            else:
                if self.position_reduction_type == "LBS":
                    # self.U = compute_position_skinning_subspace(self.model.positions, self.model.faces, k = self.position_basis_num_components)
                    pos_subspace = PositionsSubspace(self.model.positions, faces=self.model.faces,
                                                     tets=self.model.elements, num_samples=self.position_basis_num_components)
                    pos_subspace.create_basis_via_skinning_weights()
                    self.U = pos_subspace.U


                    full_global_mat = self.U.T @ full_global_mat @ self.U
                    if issparse(full_global_mat):
                        tr = full_global_mat.diagonal().sum()
                    else:
                        tr = np.trace(full_global_mat)
                    la = 1e-8 * tr / full_global_mat.shape[0]  # scale-aware lambda (to add Tikhonov regularization)

                    self.cholesky = sp.linalg.factorized(full_global_mat + la * np.eye(full_global_mat.shape[0]))

    def prepare_lbs_reduced_group(self, has_group_constraints, reduced_group, group_name,
                                  group_constraints, group_aux_size, assembly_ST, num_components, specify_verts=[]):

        if has_group_constraints and reduced_group:
            group_subspace = ConstraintsProjectionSubspace(group_name,
                                                           self.model.positions,
                                                           self.model.faces,
                                                           self.model.elements,
                                                           num_components)
            # compute skinning weights
            group_subspace.compute_skinning_weights()
            # compute constraint group mass matrix
            group_subspace.compute_constraint_mass_matrix(group_name, group_constraints, self.model.mass, group_aux_size)

            # compute lbs basis "V" for the constraint group
            group_subspace.create_basis_via_skinning_weights(self.model.positions, assembly_ST, group_constraints, group_aux_size,
                                              use_pca=False, specify_verts= specify_verts)

            if not self.reduced_position or self.position_reduction_type == "LBS":
                projecting_mat = np.einsum('ne,em->nm',assembly_ST.toarray(), group_subspace.V)
            else:
                raise ValueError(
                    "LBS reduced constraint projections can be combined only with similar position reduction.")

            # create a list of constrained elements and alpha points
            # compute interpolation solvers for lbs subspace
            group_subspace.init_constraint_group_interpolation(group_constraints,
                                                               assembly_ST,
                                                               aux_size=group_aux_size)

            print(f"Created LBS interpolation basis via skinning weights for {group_name} of size {group_subspace.V.shape}.")

            return group_subspace.alpha_points, group_subspace.sampled_constraints, projecting_mat, group_subspace.interpol_solver
        else:
            return [],[], None, []

    def prepare_lbs_reduced_verts_bending(self):
        self.bending.interpolation_alpha, self.bending.sampled_constraints, self.bending.projection_matrix, self.bending.solver_list = \
            self.prepare_lbs_reduced_group(self.model.has_verts_bending_constraints, self.bending.is_reduced,
                                       "verts_bending", self.model.verts_bending_constraints,
                                       self.bending.row_dim, self.model.verts_bending_assembly_ST,
                                       self.bending.num_components, specify_verts=self.model.verts_bending_indicies)

    def prepare_lbs_reduced_edge_spring(self):
        self.spring.interpolation_alpha, self.spring.sampled_constraints, self.spring.projection_matrix, self.spring.solver_list = \
            self.prepare_lbs_reduced_group(self.model.has_edge_spring_constraints, self.spring.is_reduced,
                                       "edge_spring", self.model.edge_spring_constraints,
                                       self.spring.row_dim, self.model.edge_spring_assembly_ST,
                                       self.spring.num_components)

    def prepare_lbs_reduced_tris_strain(self):
        self.tris_strain.interpolation_alpha, self.tris_strain.sampled_constraints, self.tris_strain.projection_matrix, self.tris_strain.solver_list = \
            self.prepare_lbs_reduced_group(self.model.has_tris_strain_constraints, self.tris_strain.is_reduced,
                                       "tris_strain", self.model.tris_strain_constraints,
                                       self.tris_strain.row_dim, self.model.tris_strain_assembly_ST,
                                       self.tris_strain.num_components)

    def prepare_lbs_reduced_tets_strain(self):
        self.tets_strain.interpolation_alpha, self.tets_strain.sampled_constraints, self.tets_strain.projection_matrix, self.tets_strain.solver_list = \
            self.prepare_lbs_reduced_group(self.model.has_tets_strain_constraints, self.tets_strain.is_reduced,
                                       "tets_strain", self.model.tets_strain_constraints,
                                       self.tets_strain.row_dim, self.model.tets_strain_assembly_ST,
                                       self.tets_strain.num_components)

    def prepare_lbs_reduced_tets_deformation_gradient(self):
        (self.tets_deformation_gradient.interpolation_alpha, self.tets_deformation_gradient.sampled_constraints,
         self.tets_deformation_gradient.projection_matrix, self.tets_deformation_gradient.solver_list) = \
            self.prepare_lbs_reduced_group(self.model.has_tets_deformation_gradient_constraints, self.tets_deformation_gradient.is_reduced,
                                       "tets_deformation_gradient", self.model.tets_deformation_gradient_constraints,
                                       self.tets_deformation_gradient.row_dim, self.model.tets_deformation_gradient_assembly_ST,
                                       self.tets_deformation_gradient.num_components)

    def prepare_snapshots_reduced_group(self, has_group_constraints, reduced_group, group_name, num_components,
                                        row_dim, assembly_ST, dir, file):
        """
            # generic constraints reduction preparation for all groups except verts bending
        Args:
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

        if has_group_constraints and reduced_group:
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

            if not self.reduced_position:
                # ST (N, ep) @ V (ep, mp, 3) --> S^T V: (N, mp, 3)
                projecting_mat = np.einsum('ne,emi->nmi',assembly_ST.toarray(), Vj)
            else:
                ## TODO: requieres test
                # U^T (r, N, 3) @ S^T (N, ep) --> U^T S^T V: (r, ep, 3)
                UtSt = np.einsum('rni,ne->rei', self.U.T, assembly_ST)
                # U^T S^T(r, ep, 3) @ Vj (ep, mp, 3) --> (r, mp, 3)
                projecting_mat = np.einsum('rei,emi->rmi', UtSt, Vj)  # TODO : check (r, m.p, 3)

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
        if self.model.has_verts_bending_constraints and self.bending.is_reduced:
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

            if not self.reduced_position:
                # ST (N, ep) @ V (ep, mp, 3) --> S^T V: (N, mp, 3)
                self.bending.projection_matrix = np.einsum('ne,emi->nmi',self.model.verts_bending_assembly_ST.toarray(), Vj)
            else:
                ## TODO: requieres test
                # U^T (r, N, 3) @ S^T (N, ep) --> U^T S^T V: (r, ep, 3)
                UtSt = np.einsum('rni,ne->rei', self.U.T, self.model.verts_bending_assembly_ST)
                # U^T S^T(r, ep, 3) @ Vj (ep, mp, 3) --> (r, mp, 3)
                self.bending.projection_matrix =  np.einsum('rei,emi->rmi', UtSt, Vj) # TODO : check (r, m.p, 3)

            PtV = Vj[self.bending.mapped_indices_Pt, :, :]   # (num_interpolation_alphas > m, mp, 3)
            PtV_T = Vj[self.bending.mapped_indices_Pt, : , :].swapaxes(0,1)   # TODO : check (m.p, m.p, 3)
            AtA = np.einsum('nai,ami->nmi',PtV_T, PtV)

            for d in range(3):
                # for each dim store [(lu_factor(AtA), At)]
                self.bending.solver_list.append([lu_factor(AtA[:, :, d]), PtV_T[:, :, d]])

    def prepare_snapshots_reduced_edge_spring(self,dir, file):
        self.spring.interpolation_alpha, self.spring.Pt, self.spring.projection_matrix, self.spring.solver_list = \
        self.prepare_snapshots_reduced_group(self.model.has_edge_spring_constraints, self.spring.is_reduced,
                                   "edge_spring", self.spring.num_components, self.spring.row_dim,
                                   self.model.edge_spring_assembly_ST,  dir, file)

    def prepare_snapshots_reduced_tris_strain(self, dir, file):
        self.tris_strain.interpolation_alpha, self.tris_strain.Pt,  self.tris_strain.projection_matrix, self.tris_strain.solver_list = \
            self.prepare_snapshots_reduced_group(self.model.has_tris_strain_constraints, self.tris_strain.is_reduced,
                                       "tris_strain", self.tris_strain.num_components, self.tris_strain.row_dim,
                                       self.model.tris_strain_assembly_ST, dir, file)

    def prepare_snapshots_reduced_tets_strain(self, dir, file):
        self.tets_strain.interpolation_alpha, self.tets_strain.Pt, self.tets_strain.projection_matrix, self.tets_strain.solver_list = \
            self.prepare_snapshots_reduced_group(self.model.has_tets_strain_constraints, self.tets_strain.is_reduced,
                                       "tets_strain", self.tets_strain.num_components, self.tets_strain.row_dim,
                                       self.model.tets_strain_assembly_ST, dir, file)

    def prepare_snapshots_reduced_tets_deformation_gradient(self, dir, file):
        self.tets_deformation_gradient.interpolation_alpha, self.tets_deformation_gradient.Pt, self.tets_deformation_gradient.projection_matrix, self.tets_deformation_gradient.solver_list = \
            self.prepare_snapshots_reduced_group(self.model.has_tets_deformation_gradient_constraints, self.tets_deformation_gradient.is_reduced,
                                       "tets_deformation_gradient", self.tets_deformation_gradient.num_components, self.tets_deformation_gradient.row_dim,
                                       self.model.tets_deformation_gradient_assembly_ST, dir, file)

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



    def prepare(self, args, store_fom_info=False, record_path=None):

        def store_assembly_matrices():
            """store a .npz contains assembly matrices for all used constraint types"""
            if store_fom_info:
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
            self.set_store_p(store_fom_info)

        if self.dirty:
            # global term computation is called every time mass matrix is changed
            self.prepare_global_matrix(args)

        if self.has_reduced_constraint_projections and not self.constraint_projection_ready:
            # called only once
            self.prepare_local_term(args)
            self.constraint_projection_ready = True

        self.set_clean()
    #-------------------------------------------------------------------------------------------------------------------
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
        p = np.zeros((ST.shape[1], 3))
        for i, c in enumerate(group_constraints):
            p[constraint_dim * i:constraint_dim * i + constraint_dim, :]  = c.get_pi(q_t)

        if self.store_stacked_projections:
            list[str(self.frame)] = p
            if self.frame == self.max_p_snapshots_num:
                np.savez(os.path.join(self.record_path, name + ".npz"), **list)
                self.set_store_p(False)
                print(f"Frame {self.frame} : FOM snapshots stored to directory", os.path.join(self.record_path, name + ".npz") )

        # update constraints projection term
        return ST @ p

    def get_group_reduced_term(self, q_t, group_constraints, constraint_dim, constrained_alphas, constrained_Pt,
                               projection_mat, solver_list):
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
            for i, alpha in enumerate(constrained_alphas):
                c = group_constraints[alpha]
                p[constraint_dim * i:constraint_dim * i + constraint_dim, :] = c.get_pi(q_t)

                def compute_rhs_column(d):
                    # in this case projection is only two dim mat
                    return projection_mat @ solve_triangular(solver_list[0], solver_list[1] @ p[:, d])

                rhs_cols = Parallel(n_jobs=3)(delayed(compute_rhs_column)(d) for d in range(3))
                temp = np.column_stack(rhs_cols)
                return temp

        else:
            raise  ValueError("Unknown constraint projection interpolation method")



    def project_to_positional_constraint_manifold(self, q_t):
        if self.model.has_positional_constraints:
            assert self.model.positional_assembly_ST is not None
            self.model.positional_stacked_p = np.zeros((self.model.positional_assembly_ST.shape[1], 3))

            for i, c in enumerate(self.model.positional_constraints):
                self.model.positional_stacked_p[i, :] = c.get_pi(q_t, self.frame)
            # if self.store_stacked_projections:
            #     np.savez(os.path.join(self.record_path, "positional_p_" + str(self.frame) + ".npz"),
            #              self.model.positional_stacked_p)
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
                                              self.bending.projection_matrix, self.bending.solver_list)
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
                                                   self.spring.solver_list)
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
                                                   self.tris_strain.solver_list)
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
                                                   self.tets_strain.solver_list)
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
                                                   self.tets_deformation_gradient.solver_list)
        return np.zeros_like(unflatten(q_t))

    def step(self, fext, num_iterations=1, use_3d_rhs_form=True):
        # global  verts_bending_p, edge_spring_p, tris_strain_p, tets_strain_p, tet_deformation_gradient_p


        N = self.model.positions.shape[0]
        self.model.positions_corrections = np.zeros_like(self.model.positions)

        dt = self.dt
        dt_inv = 1.0 / dt
        dt2 = dt * dt
        dt2_inv = 1.0 / dt2

        a = fext / self.model.mass[:, None]  # elementwise divide
        explicit = self.model.positions + dt * self.model.velocities + dt2 * a

        for v in range(self.model.positions.shape[0]):
            self.model.resolve_collision(v, explicit, self.model.positions_corrections)

        sn = flatten(explicit)

        masses = np.zeros(3 * N)
        for i in range(N):
            masses[3 * i:3 * i + 3] = dt2_inv * self.model.mass[i] * sn[3 * i:3 * i + 3]


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
                b += sum(result)  # much faster and more efficient

            else:  # can be used for full sim only
                for constraint in self.model.constraints:
                    constraint.project_wi_SiT_pi(q, b)
            b += unflatten(masses)


            if self.reduced_position:
                if self.position_reduction_type == "LBS":
                    b = self.U.T @ b.reshape(-1)
                    q = self.U @ self.cholesky(b.flatten())
                else:
                    raise ValueError("unknown position reductoin")

            else:
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
        A = sp.csc_matrix((data, (rows, cols)), shape=(3 * N, 3 * N))

        self.cholesky = sp.linalg.factorized(A)

        self.set_clean()


    def step(self, fext, num_iterations=10, use_3d_rhs_form=True, store_stacked_projections=False, record_path=None):
        # global  verts_bending_p, edge_spring_p, tris_strain_p, tets_strain_p, tet_deformation_gradient_p
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
                    if self.frame == self.max_p_snapshots_num:
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
                    if self.frame == self.max_p_snapshots_num:
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
                    if self.frame == self.max_p_snapshots_num:
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
                    if self.frame == self.max_p_snapshots_num:
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
                    if self.frame == self.max_p_snapshots_num:
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
