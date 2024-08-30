import os
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
from numpy.linalg import matrix_rank
from scipy.linalg import svd, norm, orth
import csv
import cProfile
from config.config import constProj_bases_type, constProj_support, constProj_numComponents, constProj_store_sing_val, \
                            constProj_massWeight, constProj_standarize, constProj_orthogonal, constProj_output_directory

from snapbases.nonlinear_snapshots import nonlinear_snapshots
from utils.utils import store_components, testSparsity, test_linear_dependency, store_vector

root_folder = os.getcwd()
profiler = cProfile.Profile()


class constraintsComponents:  # Components == bases
    def __init__(self):

        self.basesType = constProj_bases_type

        # position snapshots from which the components are computed are loaded and pre-processed in the following line
        self.nonlinear_snapshots = nonlinear_snapshots()

        self.numComp = constProj_numComponents  # number of bases/components
        self.support = constProj_support  # can be 'local' or 'global'

        self.storeSingVal = constProj_store_sing_val  # boolean

        self.comps = None  # bases tensor : V + average shape
        self.comps_nomean = None  # bases tensor: V
        self.weigs = None  # weights matrix
        self.ortho_comps = None  # orthogonal bases tensor
        self.largeDeforPoints = None  # points/ constraints vertices at which basis are computed
        self.largeDeforBlocks = None  # blocks containing interpolation points
        self.measures_at_largeDeforVerts = None  # singVals & resNorm computed at 'K' largest deformation verts
        self.res_at_largeDeforVerts = None  # the residual norm during PCA bases computation, all 'Kp' steps

        #self.output_components_file = output_components_file  # ??

        # TODO: file names
        self.fileNameBases = "p_nl_"
        self.fileNamenoMeanBases = None
        self.file_name_sing = "singular_values"

        # DEIM atributes
        self.deim_S = None  # All interpolation rows (complete 0< rows < e.p) used to construct P^T
        self.deim_M = None  # (P^T V)
        self.deim_res_norm = None  # norm(vk - Vc)
        self.deim_alpha = None   # Indices of interpolation blocks (0 < block < e)

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
        assert r.shape == (e, p*d)
        magnitude = r ** 2  # (e, pd)
        idx = np.argmax(magnitude.sum(axis=1))

        return idx

    def compute_components_store_singvalues(self, store_bases_dir):
        # compute_geodesic_distance = self.nonlinear_snapshots.compute_geodesic_distance
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

    def compute_nonlinearSnap_k_bases(self, writer):

        # inialized by a copy of the original snapshots tensor (F, ep, d)
        R = self.nonlinear_snapshots.snapTensor.copy()
        #  initialization
        C = []
        W = []
        S_idx = []  # stores the indices of constrained vol. verts with the largest deformation (0, e)
        S_rows = []    # stores the indices of the complete blocks in the range (0, ep)
        # add_to_indx = False   # Decide to add index to list or not

        p = self.nonlinear_snapshots.constraintsSize   # p: row size of each constraint
        e = self.nonlinear_snapshots.constraintVerts   # e: numConstraints
        self.measures_at_largeDeforVerts = np.empty((self.numComp, 6))

        res_norm = []
        for k in range(self.numComp):
            #  find the constraint index explaining the most variance across the residual animation
            idx = self.indxLargestDeformation(np.swapaxes(R, 0, 1), p, e)   # 0 <= idx < e

            # keep list of the constraints indices verts with the largest deformation  0 <= idx < ep
            S_idx.append(idx)

            sigma = []
            ck = None

            # at each largest deformation idx, a bases block of size (ep, p, 3) is computed
            for i in range(p):
                #  find linear component explaining the motion of this constraint index
                U, sing, Vt = svd(R[:, idx*p+i, :].reshape(R.shape[0], -1).T, full_matrices=False)
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
                R -= np.outer(wk, ck).reshape(R.shape)   # project out computed bases block
                #  keep list of the constraints indices of blocks with the largest deformation  0 <= idx < ep
                S_rows.append(idx * p + i)
                C.append(ck)
                W.append(wk)

            res_norm.append(norm(R))
            singList = [k, idx, sigma[0], sigma[1], sigma[2], norm(R)]
            self.measures_at_largeDeforVerts[k, :] = singList
            if self.storeSingVal:
                writer.writerow(singList)

        # print(S_idx)
        # print(res_norm)
        self.largeDeforPoints = np.array(S_idx)  # (K,)
        self.largeDeforBlocks = np.array(S_rows)  # (Kp,)
        self.res_at_largeDeforVerts = np.array(res_norm)

        if self.numComp != self.largeDeforPoints.shape[0]:
            print("Changing number of bases from ", self.numComp, 'to ', self.largeDeforPoints.shape[0])
            self.numComp = self.largeDeforPoints.shape[0]
        self.comps = np.array(C)
        # np.save('C_classes_200', self.comps)
        self.weigs = np.array(W).T

    def post_process_components(self):

        if constProj_standarize:
            # undo scaling
            self.comps /= self.nonlinear_snapshots.pre_scale_factor  # (Kp, ep, 3)

            # undo the mean-subtraction (important for bases visualisation on the full character mesh)
            self.comps += self.nonlinear_snapshots.mean[np.newaxis]  # (Kp, ep, 3) + (1, ep, 3)

            # Also for the original snaphots tensor, to be used for error measures later
            # undo scaling
            self.nonlinear_snapshots.snapTensor /= self.nonlinear_snapshots.pre_scale_factor  # (F, ep, 3)
            # undo the mean-subtraction (important for bases visualisation on the full character mesh)
            self.nonlinear_snapshots.snapTensor += self.nonlinear_snapshots.mean[np.newaxis]  # (F, ep, 3) + (1, ep, 3)

        if constProj_orthogonal:
            # orthogonalize per dimension
            for l in range(self.comps.shape[2]):
                self.comps[:, :, l] = orth(self.comps[:, :, l].T).T

        if constProj_massWeight:
            # compute M^{-1/2} U for each dimension x, y, z.
            assert self.comps.shape[1] == self.nonlinear_snapshots.invMassL.shape[0]
            self.comps *= self.nonlinear_snapshots.invMassL[:, None]

        # # only were non-zero weights
        if self.nonlinear_snapshots.rest_shape == 'first':
            self.comps_nomean = self.comps - self.nonlinear_snapshots.snapTensor[0]

        elif self.nonlinear_snapshots.rest_shape == 'average':
            self.comps_nomean = self.comps - np.mean(self.nonlinear_snapshots.snapTensor, axis=0)

    ''' 
        --- PCA Tests --- 
    '''

    def is_utmu_orthogonal(self):
        print('... testing M orthogonality, U^T M U = I (K x K) ...', end='', flush=True)
        # comps = U^T
        for l in range(self.comps.shape[2]):
            Mu_l = self.comps[:, :, l].T * self.nonlinear_snapshots.mass[:, None]   # M U
            utMu_l = np.dot(self.comps[:, :, l], Mu_l)  # U^T M U
            assert np.allclose(utMu_l, np.eye(self.comps.shape[0]))
        print('(True).')

    def test_DEIM_reconstruction_error_at_different_kp_bases(self, error_full, error_kp):
        """
        Computes re-construction error at full range of snapshots
        :param :
        error_full, error_kp : empty lists to be filled, error at all 'e' points and error at only 'k' points resp.
        :return:
        error_full = ||Vc(t) - p(t)||/(sqrt(3 ep F) ||snapshots||),
        error_kp= ||S^T Vc(t) - S^T p(t)||/(sqrt(3 kp F) ||S^T snapshots||),
                with S the 'kp' indecies of the k blocks in the 'ep' dim of V
        """

        snapshots = self.nonlinear_snapshots.snapTensor.copy()  # (F, ep, 3)
        denom = np.sqrt(3 * snapshots.shape[1] * snapshots.shape[0]) * norm(snapshots)  # sqrt(3 ep F) * ||snapshots||

        bases = np.swapaxes(self.comps, 0, 1)  # (ep, Kp, 3)

        for numBases_blocks in range(1, self.numComp+1):

            p = self.nonlinear_snapshots.constraintsSize
            S = self.deim_S[:numBases_blocks * p]  # (Kp,)
            # sqrt(3 kp F) ||S^T snapshots||
            denom_c = np.sqrt(3 * S.shape[0] * snapshots.shape[0]) * norm(snapshots[:, S, :])
            # V (ep, Kp, 3) = comp^T
            b = snapshots[:, S, :]   # S^T p(t)   (F, kp, 3)
            A = bases[S, :numBases_blocks * p, :]  # S^T V  (Kp, Kp, 3)
            c = np.empty((self.nonlinear_snapshots.frs, S.size, 3))
            z = np.empty(snapshots.shape)  # (F, ep, 3)

            # TODO: can the following part be performed with tensors operations directly?
            for f in range(self.nonlinear_snapshots.frs):
                for d in range(3):
                    c[f, :, d] = npla.solve(A[:, :, d], b[f, :, d])  # c(t) = (S^T V)^{-1} S^T p(t)  (Kp, 3) at each 'f'
                    z[f, :, d] = bases[:, :S.shape[0], d] @ c[f, :, d]  # V c(t)  (ep, 3) at each 'f'

            # TODO: try different norms
            error_full.append((norm(z - snapshots)) / denom)  # ||Vc(t) - p(t)||
            error_kp.append((norm(z[:, S, :] - snapshots[:, S, :])) / denom_c)

    def test_PCA_reconstruction_error_at_kp_bases_using_weights(self, error_full, error_kp):
        """
         Computes re-construction error at full range of snapshots
        :param error_full: error at all 'e' points
        :param error_kp: error at only 'k' points
        :return: passes filled lists
        error_full = ||WC(t) - p(t)||/(sqrt(3 ep F) ||snapshots||),
        error_kp= ||S^T WC(t) - S^T p(t)||/(sqrt(3 kp F) ||S^T snapshots||),
                with S the 'kp' indecies of the k blocks in the 'ep' dim of V
        """
        p = self.nonlinear_snapshots.constraintsSize
        e = self.nonlinear_snapshots.constraintVerts
        nfrs = self.nonlinear_snapshots.frs
        snapshots = self.nonlinear_snapshots.snapTensor.copy()  # (F, ep, 3)

        denom = np.sqrt(3 * e * p * nfrs) * norm(snapshots)

        for k in range(1, self.numComp+1):
            denom_c = np.sqrt(3 * k * p * nfrs) * norm(snapshots)
            S = self.largeDeforBlocks[:k * p]  # (Kp,)

            res = np.tensordot(self.weigs[:, :k*p], self.comps[:k*p, :, :], axes=1)
            res -= snapshots  # || WC - p(t)||  #  (F, ep, 3)

            error_full.append(norm(res)/denom)  # Kavan et.al. 2010
            error_kp.append(norm(res[:, S, :])/denom_c)

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
            s[:, i] = sing/sing.max()
        # print('min sing values over dimensions:', s[:, 0].min(), s[:, 1].min(), s[:, 2].min())
        return s

    '''
        --- DEIM/ Q-DEIM ---
    '''
    @staticmethod
    def Qdeim(Ud, pivot=None):
        n, m = Ud.shape[0], Ud.shape[1]  # (e, pKp)

        if pivot is None:
            Q, R, pivot = spla.qr(Ud.T, pivoting=True)
        else:
            U_pivoted = Ud[pivot, :]
            Q, R = spla.qr(Ud.T, pivoting=False)

        S = pivot[:m]
        M = (np.vstack((np.eye(m), spla.solve_triangular(R[:, :m], R[:, m:]).T)))
        Pinv = np.zeros(n).astype(int)
        Pinv[pivot] = range(n)
        M = M[Pinv, :]

        return S, M, pivot

    def qdeim_blocks(self):
        p = self.nonlinear_snapshots.constraintsSize
        e = self.nonlinear_snapshots.constraintVerts
        d = self.nonlinear_snapshots.dim
        K = self.numComp

        assert self.comps.shape == (K*p, e*p, d)
        bases = self.comps.copy().swapaxes(0, 1)
        bases = bases.reshape(e, p*K*p, d)

        union = []
        common = []
        for i in range(d):
            Si, Mi, pi = self.Qdeim(bases[:, :, i], None)
            if i == 0:
                union = Si
                common = Si
            else:
                common =[i for i in common if i in Si]
                union = np.union1d(union, Si)

        common_pointsFile = os.path.join(constProj_output_directory, "qdeim_common_points")
        union_pointsFile = os.path.join(constProj_output_directory, "qdeim_union_points")

        store_vector(common_pointsFile, self.nonlinear_snapshots.frs,
                     np.asarray(common).shape[0], np.asarray(common), extension='.bin')
        store_vector(union_pointsFile, self.nonlinear_snapshots.frs,
                     np.asarray(union).shape[0], np.asarray(union), extension='.bin')

    def deim_blocksForm(self):  # TODO
        """
        :return:
        """
        p = self.nonlinear_snapshots.constraintsSize
        e = self.nonlinear_snapshots.constraintVerts
        d = self.nonlinear_snapshots.dim
        K = self.numComp

        bases = self.comps.copy().swapaxes(0, 1)  # (ep, Kp, d)

        # initialization
        vk = bases[:, :p, :]  # v0  (ep, :p, 3)
        V = np.empty(vk.shape)
        # initialize selection.T mat
        Pt = []
        res = []
        e_points = []
        for k in range(K):
            if k == 0:
                r = vk
            else:
                vk = bases[:, k*p:(k+1)*p, :]  # (ep, p, d)   kth bases block
                c = np.empty(vk.shape)
                for i in range(d):
                    # V (Pt V)^{-1} Pt v_k
                    # TODO: tensor product instead?
                    c[:, :, i] = V[:, :, i] @ npla.solve(V[Pt, :, i], vk[Pt, :, i])  # (ep, kp,)@ (kp,kp)(kp,) --(ep,)xd

                r = c - vk  # error  (ep, p, d)
                res.append(norm(r))
                if np.allclose(r, np.zeros(r.shape)):
                    print("ERROR!: deim res is zero!!")
                    return

            alpha = self.indxLargestDeformation(r, p, e)
            assert alpha not in Pt
            e_points.append(alpha)

            # update selection mat with the largest deformation block in the error
            print(k, alpha)
            for m in range(p):
                Pt.append(alpha * p + m)

            if k == 0:
                V = vk
            else:
                V = np.concatenate((V, vk), axis=1)

        M = np.empty((V.shape[1], V.shape[1], d))
        Pt = np.array(Pt)
        self.deim_alpha = np.array(e_points)
        self.deim_S = Pt
        self.deim_res_norm = np.array(res)
        for i in range(d):
            M[:, :, i] = npla.inv(V[Pt, :, i])

        VM = np.zeros((e*p, V.shape[1], d))
        for i in range(d):
            VM[:, :, i] = V[:, :, i] @ M[:, :, i]

        print('Diem analysis', Pt.shape, VM.shape)
        return VM, M, Pt

    def max_raw(self, R, p):
        # R (ep, F, 3)
        magnitude = (R ** 2).sum(axis=2)
        idx = np.argmax(magnitude.sum(axis=1))
        return idx // p

    def test_deim_blocksForm_convergence(self, VM, Pt):

        snapshots = self.nonlinear_snapshots.snapTensor.copy()  # (F, ep, 3)

        denom = np.sqrt(3 * Pt.shape[0] * snapshots.shape[0])
        p = self.nonlinear_snapshots.constraintsSize
        K = VM.shape[1]//p
        error_list = []
        for k in range(K):
            Pt_ = Pt[:k*p]
            VM_ = VM[:, :k*p, :]
            for frame in range(snapshots.shape[0]):

                snap_r = snapshots[frame, Pt_, :]  # Pt p(q)

                reconstructed_snapsh = np.empty((Pt_.shape[0], 3))
                for i in range(3):
                    reconstructed_snapsh[:, i] = VM_[Pt_, :, i] @ snap_r[:, i]

                error = norm(reconstructed_snapsh - snapshots[frame, Pt_, :]) / denom
                error_list.append(error)
            print(np.mean(np.array(error_list)))
        return

    '''
        --- Store data ---
    '''

    def store_components_to_files(self, output_dir, start, end, step, bases, points, Type, fileType):
        """
        fileType can be either '.bin' or '.npy'
        """
        print('Storing bases ...', end='', flush=True)
        numframes= self.nonlinear_snapshots.frs
        numverts = self.nonlinear_snapshots.constraintVerts * self.nonlinear_snapshots.constraintsSize
        basesFile = os.path.join(output_dir, self.fileNameBases)
        # basesnoMeanFile = os.path.join(output_dir, self.fileNamenoMeanBases)
        p = self.nonlinear_snapshots.constraintsSize

        store_vector(basesFile + Type, self.nonlinear_snapshots.frs,
                    self.numComp, points, fileType)
        # store separate .bin for different numbers of components
        for k in range(start, end + 1, step):
            store_components(basesFile, numframes, k*p, numverts, 3, bases[:k*p, :, :], fileType, 'Kp')
        print('done.')

