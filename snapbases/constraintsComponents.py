import numpy.linalg as npla
from scipy.linalg import svd, norm, qr, lu_factor, lu_solve
import copy
import cProfile
import numpy as np
import matplotlib.pyplot as plt
import csv
from utils.utils import testSparsity, test_linear_dependency
from utils.support import compute_edge_incidence_matrix_on_tets, extract_sub_vertices_and_edges, extract_sub_vertices_and_tet_edges
import os
from config.config import Config_parameters
from snapbases.nonlinear_snapshots import nonlinearSnapshots
from utils.utils import store_components, store_interpol_points_vector
from utils.utils import log_time, read_sparse_matrix_from_bin
from utils.support import get_tetrahedrons_per_vert, get_triangles_per_vert, get_vertices_per_vert

import polyscope as ps

ps.init()

root_folder = os.getcwd()
profiler = cProfile.Profile()

constProj_output_directory = ""

class constraintsComponents:  # Components == bases
    def __init__(self,param: Config_parameters):
        global constProj_output_directory
        constProj_output_directory = param.constProj_output_directory

        self.basesType = ""
        self.numComp = 0  # number of bases/components
        self.support = ""  # can be 'local' or 'global'
        self.storeSingVal = False  # boolean

        # Initialize snapshots class attribute
        self.nonlinearSnapshots = nonlinearSnapshots(param)

        self.param = param
        self.comps = None  # bases tensor : V + average shape
        self.weigs = None  # weights matrix
        self.ortho_comps = None  # orthogonal bases tensor
        self.largeDeforPoints = None  # points/ constraints vertices at which basis are computed
        self.largeDeforBlocks = None  # blocks containing interpolation points
        self.measures_at_largeDeforVerts = None  # singVals & resNorm computed at 'K' largest deformation verts
        self.res_at_largeDeforVerts = None  # the residual norm during PCA bases computation, all 'Kp' steps

        self.fileNameBases = ""
        self.fileName_deim_points = ""
        self.file_name_sing = ""

        # DEIM attributes
        self.deim_interpol_verts = []   # if error is tracked in position spaces, the verts wich coneect the elements
        self.deim_alpha = None  # Indices of interpolation blocks (0 < block < e)
        # deim_alpha_ranges: a list/array to keep the number of interpolation elements in deim
        # in a fixed dim (x,y,z) bases with size (e*p, k*p) use deim_alpha[:deim_alpha_range(k)] interpolation points
        self.deim_alpha_ranges = None
        self.St = None   # differential operator that maps constraints projections to position space
        self.param = param

    def config(self, fileNameBases="p_nl_", fileName_deim_points="p_nl_interpol_points_",
               file_name_sing="_constrprojBases_pcaExtraction_singValues"):
        self.basesType = self.param.constProj_bases_type
        self.support = self.param.constProj_support  # can be 'local' or 'global'

        self.storeSingVal = self.param.constProj_store_sing_val  # boolean
        self.fileNameBases = fileNameBases
        self.fileName_deim_points = fileName_deim_points
        self.file_name_sing = file_name_sing
        self.St = read_sparse_matrix_from_bin(self.param.constProj_weightedSt)


    @staticmethod
    def project_weight(x):
        x = np.maximum(0., x)
        max_x = x.max()
        if max_x == 0:
            return x
        else:
            return x / max_x

    @staticmethod
    def indxLargestDeformation(R, p, e):
        # expected size of R (e.p, F, d)

        magnitude = (R ** 2).sum(axis=2).reshape(e, p, -1)  # (e, p, -1)
        idx = np.argmax(magnitude.sum(axis=(1, 2)))

        # temp = np.zeros(e)
        # for ind in range(e):
        #     temp[ind] = norm(R[ind*p: (ind + 1)*p, :, :])
        # idx = np.argmax(temp)
        return idx

    @staticmethod
    def indxLargestDeformation_e(r, e, p, d):
        #  R (ep, F, d)
        assert r.shape == (e, p * d)
        magnitude = r ** 2  # (e, pd)
        idx = np.argmax(magnitude.sum(axis=1))

        return idx

    @log_time(constProj_output_directory)
    def compute_components_store_singvalues(self):
        # compute_geodesic_distance = self.nonlinearSnapshots.compute_geodesic_distance
        headerSing = ['component', 'idx', 'residual_matrix_norm']
        p = self.nonlinearSnapshots.constraintsSize
        for i in range(p):
            headerSing.append('singVal'+str(i))

        file_name = os.path.join(self.param.constProj_output_directory, self.param.name +"_"+self.param.constProj_name+self.file_name_sing)
        if self.storeSingVal:
            with open(file_name + '.csv', 'w', encoding='UTF8') as singFile:
                writer = csv.writer(singFile)
                writer.writerow(headerSing)

                self.compute_nonlinearSnap_k_bases(writer)
            singFile.close()
        else:
            self.compute_nonlinearSnap_k_bases(None)

    @log_time(constProj_output_directory)
    def compute_nonlinearSnap_k_bases(self, writer=None, headerSize=0):

        # inialized by a copy of the original snapshots tensor (F, ep, d)
        R = copy.deepcopy(self.nonlinearSnapshots.snapTensor)
        #  initialization
        C = []
        W = []
        S_v_idx = []  # stores the indices of constrained vol. verts with the largest deformation (0, e)
        S_ele_idns = []  # stores the indices of the complete blocks in the range (0, ep)
        # add_to_indx = False   # Decide to add index to list or not

        p = self.nonlinearSnapshots.constraintsSize  # p: row size of each constraint
        e = self.nonlinearSnapshots.num_constained_elements  # e: numConstraints
        self.measures_at_largeDeforVerts = []
        v_count = 0
        tol = self.param.bases_R_tol
        bases_count = 0
        break_flag = False
        while norm(R) > tol:
            #  find the constraint index explaining the most variance across the residual animation
            v = np.argmax(((self.St @ np.swapaxes(R, 0, 1).reshape(e*p, -1))**2).sum(axis=1))

            if self.nonlinearSnapshots.ele_type  == "_tets":
                elems = get_tetrahedrons_per_vert([v], self.nonlinearSnapshots.tets)

            elif self.nonlinearSnapshots.ele_type  == "_tris":
                elems = get_triangles_per_vert([v], self.nonlinearSnapshots.tris)

            elif self.nonlinearSnapshots.ele_type == "_verts":
                elems = get_vertices_per_vert([v], self.nonlinearSnapshots.tris)
            else:
                print("ERROR! unknown constraints projection type")

            print("vert",v, "elements", len(elems))
            # keep list of the constraints indices verts with the largest deformation  0 <= idx < ep
            S_v_idx.append(v)

            ck = None

            # at each largest deformation idx, a bases block of size (ep, p, 3) is computed
            for idx in range(len(elems)):
                sigma = []
                S_ele_idns.append(idx)
                for i in range(p):
                    #  find linear component explaining the motion of this constraint index
                    U, sing, Vt = svd(R[:, idx * p + i, :].reshape(R.shape[0], -1).T, full_matrices=False)
                    #  R[:,idx*p+i,:].reshape(R.shape[0], -1).T is the (3,F) tensor associated to the vertex==id
                    wk = sing[0] * Vt[0, :]  # weight associated to first mode of the svd
                    # print(" wk", wk.shape)
                    sigma.append(sing[0])

                    if self.support == 'local':
                        # weight according to their projection onto the constraint set
                        # this fixes problems with negative weights and non-negativity constraints
                        wk_proj = self.project_weight(wk)
                        wk_proj_negative = self.project_weight(-wk)
                        wk = wk_proj \
                            if norm(wk_proj) > norm(wk_proj_negative) \
                            else wk_proj_negative
                        # TODO: support map for volume
                        # s = 1 - self.compute_support_map(idx, snapshots_compute_geodesic_distance,
                        #                                 self.smooth_min_dist, self.smooth_max_dist)  # (ep,)?

                    # solve for optimal component inside support map
                    # wk is (F,), R is (F, ep, 3), np.tensordot(wk, R, (0, 0)) is (ep, 3),
                    if self.support == 'local':
                        raise ValueError("Local support maps are not yet available for nonlinear-term components")
                        # TODO: depends on the support map
                        # ck = (np.tensordot(wk, R, (0, 0)) * s[:, np.newaxis]) \
                        #      / np.inner(wk, wk)
                    else:
                        ck = np.tensordot(wk, R, (0, 0)) / np.inner(wk, wk)  # (ep,3)

                    #  update residual
                    R -= np.outer(wk, ck).reshape(R.shape)  # project out computed bases block
                    print(norm(R))
                    #  keep list of the constraints indices of blocks with the largest deformation  0 <= idx < ep

                    C.append(ck)
                    W.append(wk)
                bases_count +=1
                singList = [bases_count, idx, norm(R)] # TODO: make sigma size auto in the header
                for j in range(p):
                    singList.append(sigma[j])

                self.measures_at_largeDeforVerts.append(singList)
                if self.storeSingVal:
                    writer.writerow(singList)
                if norm(R) < tol:
                    break
            v_count += 1

        # Check redundancy
        if len(S_v_idx) == len(set(S_v_idx)):
            print("PCA Large deformation verts are unique", len(S_v_idx))
        else:
            print("PCA Large deformation verts are not unique:", len(set(S_v_idx)), "points out of", len(S_v_idx))
        if len(S_ele_idns) == len(set(S_ele_idns)):
            print("PCA Large deformation elements are unique:", len(S_ele_idns))
        else:
            print("PCA Large deformation elements are not unique:", len(set(S_ele_idns)), "points out of", len(S_ele_idns))

        self.comps = np.array(C)
        self.weigs = np.array(W).T
        self.numComp = self.comps.shape[0]//p
        self.measures_at_largeDeforVerts = np.array(self.measures_at_largeDeforVerts)
        print("bases shape",self.comps.shape, "number of components", self.numComp)


    @log_time(constProj_output_directory)
    def post_process_components(self):

        if self.param.constProj_standarize:
            # undo scaling
            self.comps /= self.nonlinearSnapshots.pre_scale_factor  # (Kp, ep, 3)

            # undo the mean-subtraction (important for bases visualisation on the full character mesh)
            self.comps += self.nonlinearSnapshots.mean[np.newaxis]  # (Kp, ep, 3) + (1, ep, 3)

            # Also for the original snaphots tensor, to be used for error measures later
            # undo scaling
            self.nonlinearSnapshots.snapTensor /= self.nonlinearSnapshots.pre_scale_factor  # (F, ep, 3)
            # undo the mean-subtraction (important for bases visualisation on the full character mesh)
            self.nonlinearSnapshots.snapTensor += self.nonlinearSnapshots.mean[np.newaxis]  # (F, ep, 3) + (1, ep, 3)

        if self.param.constProj_orthogonal:
            # orthogonal per dimension
            for l in range(self.comps.shape[2]):
                self.comps[:, :, l] = qr(self.comps[:, :, l].T, mode='economic')[0].T

        if self.param.constProj_massWeight:

            # compute M^{-1/2} U for each dimension x, y, z.
            assert self.comps.shape[1] == self.nonlinearSnapshots.invMassL.shape[0]
            self.comps *= self.nonlinearSnapshots.invMassL[:, None]

            # Also for the original snapshots tensor, to be used for error measures later
            assert self.nonlinearSnapshots.snapTensor.shape[1] == self.nonlinearSnapshots.invMassL.shape[0]
            self.nonlinearSnapshots.snapTensor *= self.nonlinearSnapshots.invMassL[:, None]

        print("Post-processing, Undo standardization: ", self.param.constProj_standarize, ". Orthogonal-ized", self.param.constProj_orthogonal,
              ". Mass weighting", self.param.constProj_massWeight, ", and bases shape", self.comps.shape)

    ''' 
        --- PCA Tests --- 
    '''

    def is_utmu_orthogonal(self):
        print('... testing M orthogonality, U^T M U = I (Kp x Kp) ...', end='', flush=True)
        # comps = U^T
        for l in range(self.comps.shape[2]):
            Mu_l = self.comps[:, :, l].T * self.nonlinearSnapshots.mass[:, None]  # M U
            utMu_l = np.dot(self.comps[:, :, l], Mu_l)  # U^T M U
            assert np.allclose(utMu_l, np.eye(self.comps.shape[0]))
        print('(True).')

    def matrix_properties_test(self, interpol_kp_blocks, precondition=False):

        p = self.nonlinearSnapshots.constraintsSize
        num_interpol_points = interpol_kp_blocks.shape[0] // p
        F = self.nonlinearSnapshots.snapTensor.shape[0]
        bases = self.comps.swapaxes(0, 1)  # (ep, Kp, d)
        denom = self.nonlinearSnapshots.dim * F * num_interpol_points * p

        mat_e = np.zeros((F, num_interpol_points, 3))
        for l in range(self.nonlinearSnapshots.dim):
            for i in range(num_interpol_points):
                points = interpol_kp_blocks[:(i + 1) * p]
                JV = bases[points, :(i+1) * p, l]
                #VtJtJV = np.matmul(bases[points, :(i + 1) * p, l].T, bases[points, :(i + 1) * p, l])
                # check_matrix_properties(JV)
                lu, piv = lu_factor(JV)
                for f in range(F):
                    # x = np.linalg.solve(VtJtJV, bases[points, :(i + 1) * p, l].T
                    #                        @ self.nonlinearSnapshots.snapTensor[f, points, l])

                    x = lu_solve((lu, piv), self.nonlinearSnapshots.snapTensor[f, points, l])
                    r = bases[:, :(i + 1) * p, l] @ x - self.nonlinearSnapshots.snapTensor[f, :, l]
                    mat_e[f, i, l] = norm(r) / denom / norm(self.nonlinearSnapshots.snapTensor[f, :, l])

        return mat_e

    def deim_train_constructed(self, r):

        p = self.nonlinearSnapshots.constraintsSize
        F, ep, _ = self.nonlinearSnapshots.snapTensor.shape

        V_r = self.comps.swapaxes(0, 1)[:, :r*p, :]  # (ep, rp, 3)
        Pt = self.deim_alpha[:self.deim_alpha_ranges[r-1]]

        reconstructed = np.zeros((F, ep, 3))

        for l in range(3):
            u, piv = lu_factor(V_r[Pt, :, l].T @ V_r[Pt, :, l])
            for f in range(F):
                reconstructed[f, :, l] = V_r[:, :, l] @ lu_solve((u, piv), V_r[Pt, :, l].T @ self.nonlinearSnapshots.snapTensor[f, Pt, l])

        return reconstructed

    def deim_test_constructed(self, r):

        p = self.nonlinearSnapshots.constraintsSize
        F, ep, _ = self.nonlinearSnapshots.test_snapTensor.shape

        V_r = self.comps.swapaxes(0, 1)[:, :r*p, :]  # (ep, rp, 3)
        Pt = self.deim_alpha[:self.deim_alpha_ranges[r-1]]

        reconstructed = np.zeros((F, ep, 3))

        for l in range(3):
            u, piv = lu_factor(V_r[Pt, :, l].T @ V_r[Pt, :, l])
            for f in range(F):
                reconstructed[f, :, l] = V_r[:, :, l] @ lu_solve((u, piv), V_r[Pt, :, l].T @ self.nonlinearSnapshots.test_snapTensor[f, Pt, l])

        return reconstructed

    def reconstruct(self, rp):
        """
        Reconstructs the data using the reduced basis.
        :param f: Original tensor of shape (F, ep, 3)
        :param V_r: Basis tensor of shape (ep, rp, 3)
        :return: Reconstructed tensor of shape (F, ep, 3)
        """
        f = self.nonlinearSnapshots.snapTensor
        F, ep, _ = f.shape
        V_r = self.comps.swapaxes(0, 1)[:, :rp, :]  # (ep, rp, 3)

        reconstructed = np.zeros((F, ep, 3))

        for i in range(3):  # For each component (x, y, z)
            # Project the snapshots onto the reduced basis
            coefficients = f[:, :, i] @ V_r[:, :, i]  # Shape (F, rp)
            reconstructed[:, :, i] = coefficients @ V_r[:, :, i].T  # Shape (F, ep)

        return reconstructed

    @staticmethod
    def frobenius_error(f, f_reconstructed):
        """
        Computes the Frobenius norm of the reconstruction error.
        """
        error = f - f_reconstructed
        return np.linalg.norm(error)

    @staticmethod
    def relative_error_per_component(f, f_reconstructed):
        """
        Computes the relative error for each component.
        """
        relative_errors = []  # Relative errors for (x, y, z)

        for i in range(3):
            norm_original = np.linalg.norm(f[:, :, i])
            norm_error = np.linalg.norm(f[:, :, i] - f_reconstructed[:, :, i])
            relative_errors.append(norm_error / norm_original)

        return relative_errors

    @staticmethod
    def max_pointwise_error(f, f_reconstructed):
        """
        Computes the maximum point-wise error.
        :param f: Original tensor (T, N, 3)
        :param f_reconstructed: Reconstructed tensor (T, N, 3)
        :return: Maximum point-wise reconstruction error
        """
        error = np.abs(f - f_reconstructed)
        return np.max(error)/np.max(f)

    def test_basesSingVals(self):
        """
        Computes normalized singular values along all Kp vectors on the already fully-extracted PCA bases
        :return:
        s = [sx, sy, sz] normalized for each dim separately
        """
        bases = self.comps.copy()  # (Kp, ep, 3)
        s = np.empty((bases.shape[0], 3))
        for i in range(3):
            U, sing, Vt = svd(bases[:, :, i], full_matrices=False)
            s[:, i] = sing / sing.max()

        # print('min sing values over dimensions:', s[:, 0].min(), s[:, 1].min(), s[:, 2].min())
        return s

    @log_time(constProj_output_directory)
    def store_components_to_files(self, start, end, step, fileType):
        """
        fileType can be either '.bin' or '.npy'
        """
        print('Storing bases ...', end='', flush=True)
        numframes = self.nonlinearSnapshots.frs
        numverts = self.nonlinearSnapshots.num_constained_elements * self.nonlinearSnapshots.constraintsSize
        basesFile = os.path.join(constProj_output_directory, self.fileNameBases)
        pointsFile = os.path.join(constProj_output_directory, self.fileName_deim_points)
        vertsFile = os.path.join(constProj_output_directory, "corrVerts")
        p = self.nonlinearSnapshots.constraintsSize

        # store separate .bin for different numbers of components
        for k in range(start, end + 1, step):
            store_components(basesFile, numframes, k * p, numverts, 3, self.comps[:k * p, :, :], fileType, 'Kp')

            store_interpol_points_vector(pointsFile, self.nonlinearSnapshots.frs, k, self.deim_alpha[:self.deim_alpha_ranges[k - 1]], fileType)

            store_interpol_points_vector(vertsFile, self.nonlinearSnapshots.frs, k,
                                         self.deim_interpol_verts[:k], fileType)

        print('done.')

    '''
        --- DEIM/ Q-DEIM ---
    '''
    @log_time(constProj_output_directory)
    def deim_blocksForm(self, error_in_pos_space=False):
        """
        :return:
        """
        p = self.nonlinearSnapshots.constraintsSize
        e = self.nonlinearSnapshots.num_constained_elements
        d = self.nonlinearSnapshots.dim
        K = self.numComp

        bases = self.comps.swapaxes(0, 1)  # (ep, Kp, d)
        if error_in_pos_space:
            if self.nonlinearSnapshots.ele_type not in ["_tets", "_tris", "_verts"] :
                print("ERROR! Unknown constained elements type in deim")
                return

        # initialization
        vk = bases[:, :p, :]  # v0  (ep, :p, 3)
        V = np.empty(vk.shape)
        # initialize selection.T mat
        Pt = []
        e_points = []
        e_jump =[]
        e_range = []
        for k in range(K):
            if k == 0:
                if error_in_pos_space:
                    r = self.St @ vk.reshape(vk.shape[0], -1)
                else:
                    r = vk
            else:
                vk = bases[:, k * p:(k + 1) * p, :]  # (ep, p, d)   kth bases block
                c = np.empty(vk.shape)
                if error_in_pos_space:
                    for i in range(d):
                        # V (Pt V)^{-1} Pt v_k
                        # check_matrix_properties(V[Pt, :, i].T @ V[Pt, :, i])
                        # In this case we solve for the normal equation because we have more rows than cols
                        c[:, :, i] = V[:, :, i] @ npla.solve(V[Pt, :, i].T @ V[Pt, :, i], V[Pt, :, i].T @ vk[Pt, :, i])  # (ep, kp,)@ (kp,) --(ep,)xd

                    r = c - vk  # residual in constraint projection space  (ep, p, d)
                    r = self.St @ r.reshape(r.shape[0], -1)   # residual in position space (|V|, p*d)
                else:
                    for i in range(d):
                        # V (Pt V)^{-1} Pt v_k
                        # check_matrix_properties(V[Pt, :, i])
                        # In this case we solve for a square matrix
                        c[:, :, i] = V[:, :, i] @ npla.solve(V[Pt, :, i], vk[Pt, :, i])  # (ep, kp,)@ (kp,kp)(kp,) --(ep,)xd

                    r = c - vk  # residual in constraint projection space  (ep, p, d)
                if np.allclose(r, np.zeros(r.shape)):
                    print("ERROR!: deim res is zero!!")
                    return

            if error_in_pos_space:
                v_interpolate = np.argmax((r ** 2).sum(axis=1))
                self.deim_interpol_verts.append(v_interpolate)   # list of verts that showed largest errors
                if self.nonlinearSnapshots.ele_type == "_tets":
                    alpha_list = get_tetrahedrons_per_vert([v_interpolate], self.nonlinearSnapshots.tets)
                elif self.nonlinearSnapshots.ele_type == "_tris":
                    alpha_list = get_triangles_per_vert([v_interpolate], self.nonlinearSnapshots.tris)
                elif self.nonlinearSnapshots.ele_type == "_verts":
                    alpha_list = get_vertices_per_vert([v_interpolate], self.nonlinearSnapshots.tris)

                jump = 0
                for al in range(len(alpha_list)):
                    alpha = alpha_list[al]
                    if alpha not in e_points and jump < self.param.deim_ele_per_vert:
                        jump +=1
                        e_points.append(alpha)
                        # update selection mat with all elements that show largest deformation in position space
                        print(k, alpha)
                        for m in range(p):
                            Pt.append(alpha * p + m)
                e_jump.append(jump)
                e_range.append(sum(np.asarray(e_jump)))
            else:
                alpha = self.indxLargestDeformation(r, p, e)
                assert alpha not in e_points
                e_points.append(alpha)
                # update selection mat with the largest deformation block in the error
                print(k, alpha)
                for m in range(p):
                    Pt.append(alpha * p + m)
                e_jump.append(1)
                e_range.append(sum(np.asarray(e_jump)))
            if k == 0:
                V = vk
            else:
                V = np.concatenate((V, vk), axis=1)

        self.deim_alpha = np.array(e_points)
        self.deim_alpha_ranges = np.array(e_range)
        self.deim_interpol_verts = np.array(self.deim_interpol_verts)
        print("Deim interpolation used", self.deim_alpha.shape[0], "constrained elements")


    '''
        --- Results/ Tests ---
    '''
    @log_time(constProj_output_directory)
    def tets_plots_deim(self):
        """
        Plots different reconstruction errors for varying reduction dimensions "r".
        :param f: Original tensor (T, N, 3)
        :param V_f: Basis tensor (N, max_r, 3)
        :param max_r: Maximum reduction dimension (number of modes)
        """
        constProj_output_directory = self.param.constProj_output_directory

        def run_tests():
            k = self.numComp
            p = self.nonlinearSnapshots.constraintsSize
            rp_values = range(p, k * p + 1, p)
            r_values = range(1, k + 1)

            # PCA tests --------------------------------------------------------------------------------------------------------
            plt.figure('Error measures for PCA.', figsize=(20, 10))

            store_kp_singVals = True

            rows = 1
            cols = 3
            plt.subplot(rows, cols, 1)

            # singular values at 'K' largest deformation blocks during PCA bases extarction
            # singVals starts from entry 4 in the measures_at_largeDeforVerts array
            mark = ['bo', 'ro', 'go']
            for i in range(p):
                plt.plot(r_values, self.measures_at_largeDeforVerts[:, 3 + i] /
                         self.measures_at_largeDeforVerts[:, 3 + i].max(), mark[i], ls='-.',
                         label=f'$\sigma_{{{i}}}$')

            plt.legend(loc='upper center')
            plt.xlabel('Reduction Dimension (r)')
            plt.ylabel('Normalized $\sigma$')
            plt.title("singVals at larg_deformation points")
            # plt.yscale("log")
            # plt.xticks(np.arange(1, k+1, 1))
            plt.legend()

            plt.subplot(rows, cols, 2)
            # residual_norm values at 'K' largest deformation blocks: norm(R) is expected to be the 3rd entry
            plt.plot(r_values, self.measures_at_largeDeforVerts[:, 2], 'rv', ls='-',
                     label='$\| R_{pca} \|_F$ blocks')
            plt.legend(loc='upper center')
            plt.xlabel('Reduction Dimension (r)')
            plt.ylabel('Fro. nom')
            plt.title("norm(R) at K PCA bases extraction")
            # plt.yscale("log")
            # plt.xticks(np.arange(1, k + 1, 1))
            plt.legend()

            plt.subplot(rows, cols, 3)
            # singular vals for the bases over full Kp range
            if store_kp_singVals:
                header_ = ['i', 'x', 'y', 'z']
                file_name_ = os.path.join(constProj_output_directory,
                                          self.param.name + "_constrprojBases_xyz_fullBasesRange_Kp_singVals")
                with open(file_name_ + '.csv', 'w', encoding='UTF8') as dataFile_:
                    writer_ = csv.writer(dataFile_)
                    writer_.writerow(header_)

                    s = self.test_basesSingVals()
                    for row in range(s.shape[0]):
                        writer_.writerow([row + 1, s[row, 0], s[row, 1], s[row, 2]])

                    for row in range(s.shape[0]):
                        writer_.writerow([row, s[row, :]])
                dataFile_.close()
            else:
                s = self.test_basesSingVals()

            values = range(1, k * p + 1, 1)
            plt.plot(values, s[:, 0], 'bo', ls='--', label='$\sigma_x$')
            plt.plot(values, s[:, 1], 'ro', ls='--', label='$\sigma_y$')
            plt.plot(values, s[:, 2], 'go', ls='--', label='$\sigma_z$')
            plt.legend(loc='upper center')
            plt.xlabel('Reduction Dimension (r)')
            plt.ylabel('Fro. nom')
            plt.title("Normalized singVal(bases), full Kp range")
            # plt.yscale("log")
            # plt.xticks(values)
            plt.legend()
            fig_name = os.path.join(constProj_output_directory, 'constrprojBases_pca_extraction_tests')
            plt.savefig(fig_name)
            # End of PCA tests -------------------------------------------------------------------------------------------------

            # After post-process tests
            testSparsity(self.comps)
            test_linear_dependency(self.comps, 3,
                                   self.numComp * self.nonlinearSnapshots.constraintsSize)

            if self.param.constProj_orthogonal:
                self.is_utmu_orthogonal()  # test U^T M U = I (Kp x Kp)

            # DEIM tests -------------------------------------------------------------------------------------------------------

            frobenius_errors = []
            max_errors = []
            relative_errors_x = []
            relative_errors_y = []
            relative_errors_z = []
            best_num_element_to_plot = 0
            f = self.nonlinearSnapshots.snapTensor

            header = ['numPoints', 'fro_error', 'max_err', 'relative_errors_x', 'relative_errors_y',
                      'relative_errors_z']

            file_name = os.path.join(self.param.constProj_output_directory, "deim_convergence_tests")
            with open(file_name + '.csv', 'w', encoding='UTF8') as dataFile:
                writer = csv.writer(dataFile)
                writer.writerow(header)
                for r in r_values:
                    # Reconstruct the tensor for the current r
                    f_reconstructed = self.deim_constructed(r)

                    # Compute various errors
                    fro_error = self.frobenius_error(f, f_reconstructed)
                    max_err = self.max_pointwise_error(f, f_reconstructed)
                    rel_errors = self.relative_error_per_component(f, f_reconstructed)

                    # Store errors
                    frobenius_errors.append(fro_error)
                    max_errors.append(max_err)
                    relative_errors_x.append(rel_errors[0])
                    relative_errors_y.append(rel_errors[1])
                    relative_errors_z.append(rel_errors[2])

                    writer.writerow([r, fro_error, max_err, rel_errors[0], rel_errors[1], rel_errors[2]])
            dataFile.close()
            # Plot Frobenius and inf norm error
            plt.figure('Error measures for DEIM ', figsize=(20, 10))

            plt.subplot(1, 2, 1)
            plt.plot(frobenius_errors, label='Frobenius Error', marker='o')
            plt.plot(r_values, max_errors, label='Inf Error', marker='o')
            plt.xlabel('Reduction Dimension (r)')
            plt.ylabel('Error')
            plt.title('Frobenius Norm')
            plt.yscale("log")
            # Set x-ticks to integers only
            # plt.xticks(np.arange(1, k + 1, 1)) # range 0 <= r <= numComponents
            plt.legend()

            # Plot Relative Errors for each component (x, y, z)
            plt.subplot(1, 2, 2)
            relative_error = np.sum(np.array([relative_errors_x, relative_errors_y, relative_errors_z]), axis=0)
            plt.plot(r_values, relative_errors_x, label='Relative Error X', marker='o')
            plt.plot(r_values, relative_errors_y, label='Relative Error Y', marker='x')
            plt.plot(r_values, relative_errors_z, label='Relative Error Z', marker='s')
            plt.plot(r_values, relative_error, label='sumRelative Error', marker='v')
            plt.xlabel('Reduction Dimension (r)')
            plt.ylabel('Relative Error')
            plt.title('Relative Errors per Component (X, Y, Z)')
            plt.yscale("log")
            # plt.xticks(np.arange(1, k+1, 1))
            plt.legend()

            # plt.tight_layout()
            fig_name = os.path.join(constProj_output_directory, 'constrproj_deim_reconstruction_norms_tests')
            plt.savefig(fig_name)

            plt.figure('Number of constrained elements in DEIM ', figsize=(20, 10))
            plt.subplot(1, 1, 1)
            plt.plot(self.deim_alpha_ranges, 'bo', ls='--', label=' 0 < elements < e')
            plt.xlabel('Reduction Dimension (r)')
            plt.ylabel('number of elements')
            plt.title('Number of constrained elements in DEIM ')

            fig_name = os.path.join(constProj_output_directory, 'deim_numberOfElements')
            plt.legend()
            plt.savefig(fig_name)

            if self.param.visualize_deim_elements:
                self.visualize_interpolation_elements(self.param.visualize_deim_elements_at_K,
                                                 constProj_output_directory)
            plt.close()

        run_tests()
        # End of DEIM tests ------------------------------------------------------------------------------------------------

        # plt.show()

    def visualize_interpolation_elements(self, visualize_deim_elements_at_K,
                                         constProj_output_directory, ele_color=(0.5, 0.8, 0.5), num_frames=30,
                                         file_prefix="frame"):
        """
        Highlights specific elements (vertices, tetrahedra, faces) in a tetrahedral mesh using Polyscope.

        Parameters:
        - vertices: np.ndarray, array of vertex positions.
        - tets: np.ndarray, array of tetrahedral indices.
        - highlight_verts: list[int], indices of vertices to highlight.
        - highlight_tets: list[int], indices of tetrahedra to highlight.
        - highlight_faces: list[tuple], specific faces (triplets of vertex indices) to highlight.
        """

        deim_verts = self.deim_interpol_verts[:visualize_deim_elements_at_K]
        highlight_elements = self.deim_alpha[
                             :self.deim_alpha_ranges[visualize_deim_elements_at_K - 1]]
        highlight_type = self.nonlinearSnapshots.ele_type

        # Register the mesh
        ps.register_surface_mesh("Tet Mesh", self.nonlinearSnapshots.verts,
                                 self.nonlinearSnapshots.tris,
                                 transparency=0.1, color=(0.89, 0.807, 0.565))
        ps.register_point_cloud("deim Vertices", self.nonlinearSnapshots.verts[deim_verts], enabled=True,
                                color=(0.9, 0.1, 0.25), radius=0.008)

        # Highlight vertices
        if highlight_type == "_verts":
            ps.register_point_cloud("Highlighted Vertices",
                                    self.nonlinearSnapshots.verts[highlight_elements],
                                    enabled=True, color=ele_color)

        # Highlight tetrahedra
        elif highlight_type == "_tets":
            ps.register_volume_mesh("Highlighted Tets", self.nonlinearSnapshots.verts,
                                    self.nonlinearSnapshots.tets[highlight_elements], transparency=0.8,
                                    color=ele_color)

        # Highlight faces
        elif highlight_type == "_tris":
            ps.register_surface_mesh("Highlighted Faces", self.nonlinearSnapshots.verts,
                                     self.nonlinearSnapshots.tris[highlight_elements], transparency=0.8,
                                     color=ele_color)

        # Highlight edges
        elif highlight_type == "_triEdges":
            edges = self.nonlinearSnapshots.edges[highlight_elements]
            sub_verts, sub_edges = extract_sub_vertices_and_edges(self.nonlinearSnapshots.verts, edges,
                                                                  transparency=0.8,
                                                                  color=ele_color)
            ps.register_curve_network("Highlighted Tri- Edges", sub_verts, sub_edges)

        elif highlight_type == "_tetEdges":
            # check if required!
            edges = compute_edge_incidence_matrix_on_tets(self.nonlinearSnapshots.tets)[highlight_elements]
            sub_verts, sub_edges = extract_sub_vertices_and_tet_edges(self.nonlinearSnapshots.verts, edges)
            ps.register_curve_network("Highlighted Tet-Edges", sub_verts, sub_edges)

        # ps.show()
        output_dir = os.path.join(constProj_output_directory, "rotation_scene_snapshots")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Compute the bounding box of the vertices
        min_corner = np.min(self.nonlinearSnapshots.verts, axis=0)
        max_corner = np.max(self.nonlinearSnapshots.verts, axis=0)
        center = (min_corner + max_corner) / 2
        bounding_box_size = np.linalg.norm(max_corner - min_corner)

        # Determine camera distance (e.g., 2x bounding box size for full view)
        camera_distance = 1.1 * bounding_box_size
        ps.set_ground_plane_mode("none")

        global angle, frame
        angle = 360.0 / num_frames
        frame = 1

        def callback():
            # Rotate the view incrementally
            global angle, frame
            angle += angle
            camera_position = (
                center[0] + camera_distance * np.sin(np.radians(angle)),
                center[1],
                center[2] + camera_distance * np.cos(np.radians(angle)),
            )
            target_position = center  # Look at the center of the bounding box
            ps.look_at(camera_position, target_position)

            if frame <= num_frames:
                # Capture the screenshot
                filename = os.path.join(output_dir,
                                        self.param.name + "_" + self.param.constProj_name + "_" + f"{file_prefix}_{frame:03d}.png")
                ps.screenshot(filename, transparent_bg=False)
                frame += 1
            else:
                print(f"Reached frame {frame + 1}, closing Polyscope.")
                ps.unshow()

        # Update the Polyscope viewer
        ps.set_user_callback(callback)
        ps.show()

        print(f"Captured {num_frames} frames in {output_dir}")




