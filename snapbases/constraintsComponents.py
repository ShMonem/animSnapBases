import os
import numpy as np
import numpy.linalg as npla
from scipy.linalg import svd, norm, qr, lu_factor, lu_solve
import copy
import csv
import cProfile
from config.config import constProj_bases_type, constProj_support, constProj_numComponents_verts, constProj_store_sing_val, \
    constProj_massWeight, constProj_standarize, constProj_orthogonal, constProj_output_directory, constProj_weightedSt,\
    constProj_snapshots_type

from snapbases.nonlinear_snapshots import nonlinearSnapshots
from utils.utils import store_components, store_interpol_points_vector
from utils.utils import log_time, read_sparse_matrix_from_bin
from utils.support import find_tetrahedrons_with_vertices
root_folder = os.getcwd()
profiler = cProfile.Profile()


class constraintsComponents:  # Components == bases
    def __init__(self):

        self.basesType = ""
        self.numComp = 0  # number of bases/components
        self.nvu = 0  # max number of vertecies around which we compute bases
        self.support = ""  # can be 'local' or 'global'
        self.storeSingVal = False  # boolean

        # Initialize snapshots class attribute
        self.nonlinearSnapshots = nonlinearSnapshots()

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

        # DEIM atributes
        # self.deim_S = None  # All interpolation rows (complete 0< rows < e.p) used to construct P^T
        # self.deim_M = None  # (P^T V)
        # self.deim_res_norm = None  # norm(vk - Vc)
        self.deim_alpha = None  # Indices of interpolation blocks (0 < block < e), error measure in constained elements space
        self.St = None   # differential operator that maps constarints projections to position space

    def config(self, fileNameBases="p_nl_", fileName_deim_points="p_nl_interpol_points_",
               file_name_sing="pca_singValues"):

        self.basesType = constProj_bases_type
        self.nvu = constProj_numComponents_verts  # number of bases/components
        self.support = constProj_support  # can be 'local' or 'global'

        self.storeSingVal = constProj_store_sing_val  # boolean
        self.fileNameBases = fileNameBases
        self.fileName_deim_points = fileName_deim_points
        self.file_name_sing = file_name_sing

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
    def compute_components_store_singvalues(self, store_bases_dir):
        # compute_geodesic_distance = self.nonlinearSnapshots.compute_geodesic_distance
        headerSing = ['component', 'idx', 'singVal1', 'singVal2', 'singVal3', 'residual_matrix_norm']

        file_name = os.path.join(store_bases_dir, self.file_name_sing)
        if self.storeSingVal:
            with open(file_name + "_K_" + str(self.numComp) + '.csv', 'w', encoding='UTF8') as singFile:
                writer = csv.writer(singFile)
                writer.writerow(headerSing)

                self.compute_nonlinearSnap_k_bases(writer)
            singFile.close()
        else:
            self.compute_nonlinearSnap_k_bases(None)

    @log_time(constProj_output_directory)
    def compute_nonlinearSnap_k_bases(self, writer):

        # inialized by a copy of the original snapshots tensor (F, ep, d)
        R = copy.deepcopy(self.nonlinearSnapshots.snapTensor)
        #  initialization
        C = []
        W = []
        S_v_idx = []  # stores the indices of constrained vol. verts with the largest deformation (0, e)
        S_ele_idns = []  # stores the indices of the complete blocks in the range (0, ep)
        # add_to_indx = False   # Decide to add index to list or not

        p = self.nonlinearSnapshots.constraintsSize  # p: row size of each constraint
        e = self.nonlinearSnapshots.constraintVerts  # e: numConstraints
        #self.measures_at_largeDeforVerts = np.empty((self.numComp, 6))
        self.St = read_sparse_matrix_from_bin(constProj_weightedSt)

        v_count = 0
        tol = 1e-40
        bases_count = 0
        while v_count < self.nvu and norm(R) > tol:
            #  find the constraint index explaining the most variance across the residual animation
            #idx = self.indxLargestDeformation(np.swapaxes(R, 0, 1), p, e)  # 0 <= idx < e
            v = np.argmax(((self.St @ np.swapaxes(R, 0, 1).reshape(e*p, -1))**2).sum(axis=1))

            if constProj_snapshots_type == "tetstrain":
                elems = find_tetrahedrons_with_vertices([v], self.nonlinearSnapshots.tets)
            else:
                print("ERROR! not yet implemented") # TODO

            print("vert",v, "elements", len(elems))
            # keep list of the constraints indices verts with the largest deformation  0 <= idx < ep
            S_v_idx.append(v)
            sigma = []
            ck = None

            # at each largest deformation idx, a bases block of size (ep, p, 3) is computed
            for idx in range(len(elems)):
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
                        dummyindx = 0
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

                #self.measures_at_largeDeforVerts[k, :] = singList
                if self.storeSingVal:
                    writer.writerow(singList)

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
        print("bases shape",self.comps.shape, "number of components", self.numComp)



    @log_time(constProj_output_directory)
    def post_process_components(self):

        if constProj_standarize:
            # undo scaling
            self.comps /= self.nonlinearSnapshots.pre_scale_factor  # (Kp, ep, 3)

            # undo the mean-subtraction (important for bases visualisation on the full character mesh)
            self.comps += self.nonlinearSnapshots.mean[np.newaxis]  # (Kp, ep, 3) + (1, ep, 3)

            # Also for the original snaphots tensor, to be used for error measures later
            # undo scaling
            self.nonlinearSnapshots.snapTensor /= self.nonlinearSnapshots.pre_scale_factor  # (F, ep, 3)
            # undo the mean-subtraction (important for bases visualisation on the full character mesh)
            self.nonlinearSnapshots.snapTensor += self.nonlinearSnapshots.mean[np.newaxis]  # (F, ep, 3) + (1, ep, 3)

        if constProj_orthogonal:
            # orthogonal per dimension
            for l in range(self.comps.shape[2]):
                self.comps[:, :, l] = qr(self.comps[:, :, l].T, mode='economic')[0].T

        if constProj_massWeight:

            # compute M^{-1/2} U for each dimension x, y, z.
            assert self.comps.shape[1] == self.nonlinearSnapshots.invMassL.shape[0]
            self.comps *= self.nonlinearSnapshots.invMassL[:, None]

            # Also for the original snapshots tensor, to be used for error measures later
            assert self.nonlinearSnapshots.snapTensor.shape[1] == self.nonlinearSnapshots.invMassL.shape[0]
            self.nonlinearSnapshots.snapTensor *= self.nonlinearSnapshots.invMassL[:, None]

        print("Post-processing, Undo standardization: ", constProj_standarize, ". Orthogonal-ized", constProj_orthogonal,
              ". Mass weighting", constProj_massWeight, ", and bases shape", self.comps.shape)

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

    def deim_constructed(self, rp):

        p = self.nonlinearSnapshots.constraintsSize
        f = self.nonlinearSnapshots.snapTensor
        F, ep, _ = f.shape

        V_r = self.comps.swapaxes(0, 1)[:, :rp, :]  # (ep, rp, 3)
        Pt = self.deim_S[:rp]

        reconstructed = np.zeros((F, ep, 3))

        for l in range(3):
            u, piv = lu_factor(V_r[Pt, :, l])
            for f in range(F):
                reconstructed[f, :, l] = V_r[:, :, l] @ lu_solve((u, piv), self.nonlinearSnapshots.snapTensor[f, Pt, l])

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
        return np.linalg.norm(error)/norm(f)

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

    def test_basesSingVals(self, writer=None):
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

            if writer is not None:
                writer.writerow(s[:, i])
        # print('min sing values over dimensions:', s[:, 0].min(), s[:, 1].min(), s[:, 2].min())
        return s

    @log_time(constProj_output_directory)
    def store_components_to_files(self, output_dir, start, end, step, bases, points, fileType):
        """
        fileType can be either '.bin' or '.npy'
        """
        print('Storing bases ...', end='', flush=True)
        numframes = self.nonlinearSnapshots.frs
        numverts = self.nonlinearSnapshots.constraintVerts * self.nonlinearSnapshots.constraintsSize
        basesFile = os.path.join(output_dir, self.fileNameBases)
        pointsFile = os.path.join(output_dir, self.fileName_deim_points)
        p = self.nonlinearSnapshots.constraintsSize

        store_interpol_points_vector(pointsFile, self.nonlinearSnapshots.frs, self.numComp, points, fileType)
        # store separate .bin for different numbers of components
        for k in range(start, end + 1, step):
            store_components(basesFile, numframes, k * p, numverts, 3, bases[:k * p, :, :], fileType, 'Kp')
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
        e = self.nonlinearSnapshots.constraintVerts
        d = self.nonlinearSnapshots.dim
        K = self.numComp
        St = None
        bases = self.comps.swapaxes(0, 1)  # (ep, Kp, d)
        if error_in_pos_space:
            if constProj_snapshots_type != "tetstrain":
                print("ERROR! Unknown constaints projection type in deim") # TODO generalize
                return
            St = read_sparse_matrix_from_bin(constProj_weightedSt)
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
                    r = St @ vk.reshape(vk.shape[0], -1)
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
                    r = St @ r.reshape(r.shape[0], -1)   # residual in position space (|V|, p*d)
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
                alpha_list = find_tetrahedrons_with_vertices([v_interpolate], self.nonlinearSnapshots.tets)
                jump = 0
                for al in range(len(alpha_list)):
                    alpha = alpha_list[al]
                    if alpha not in e_points:
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
        print("Deim interpolation used", self.deim_alpha.shape[0], "constrained elements", self.deim_alpha)
        print(len(e_jump), e_jump)
        print(len(e_range), e_range)
