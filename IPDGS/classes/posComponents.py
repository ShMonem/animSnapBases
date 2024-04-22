
import os
import csv
import h5py
from numpy import maximum, argmax, clip, save, tensordot, inner, outer, array, newaxis, empty, zeros, dot, eye, sum,\
                  sqrt, errstate, maximum, allclose, mean
from scipy.linalg import svd, norm, cho_factor, cho_solve, cholesky, orth
from utils.utils import store_components, testSparsity, test_linear_indpendency
from IPDGS.classes.posSnapshots import posSnapshots
from IPDGS.config.config import vertPos_bases_type, store_vertPos_PCA_sing_val, vertPos_rest_shape, vertPos_maxFrames,\
                                vertPos_numFrames, vertPos_numComponents, vertPos_smooth_min_dist, \
                                vertPos_smooth_max_dist, vertPos_masses_file, q_standarize, q_massWeight, \
                                q_orthogonal, q_support, q_store_sing_val, input_animation_dir, \
                                vertPos_output_directory, splocs_max_itrs, \
                                splocs_admm_num_itrs, splocs_lambda, splocs_rho


class posComponents:  # Components == bases
    def __init__(self):

        # bases type, can only be 'PCA' and 'SPLOCS'
        self.basesType = vertPos_bases_type
        assert self.basesType == 'PCA' or self.basesType == 'SPLOCS'

        # position snapshots from which the components are computed are loaded and pre-processed if required
        self.pos_snapshots = posSnapshots(input_animation_dir, vertPos_rest_shape, vertPos_masses_file,
                                          q_standarize, q_massWeight)

        self.numComp = vertPos_numComponents  # max number of components
        self.support = q_support  # can be 'local' or 'global'

        self.storeSingVal = store_vertPos_PCA_sing_val  # boolean

        self.comps = None  # bases tensor : U + average shape
        self.weigs = None  # weights matrix
        self.ortho_comps = None  # orthogonal bases tensor
        self.smooth_min_dist = vertPos_smooth_min_dist  # minimum geodesic distance for support map, d_min_in paper
        self.smooth_max_dist = vertPos_smooth_max_dist
        self.output_components_file = "components.h5"
        self.output_animation_file = "animations.h5"

        self.fileNameBases = None
        # self.fileNamenoMeanBases = None
        # self.file_name_sing = None

        self.fileNameBases = "using_F_"
        # self.file_name_sing = vertPos_singVals_dir

    @staticmethod
    def project_weight(x):
        x = maximum(0., x)
        max_x = x.max()
        if max_x == 0:
            return x
        else:
            return x / max_x

    @staticmethod
    def compute_support_map(idx, geodesics, min_dist, max_dist):  # In paper Eq(6) before multiplying with lambda
        phi = geodesics(idx)
        return (clip(phi, min_dist, max_dist) - min_dist) \
               / (max_dist - min_dist)

    def extract_k_components(self, residual_init, snapshots_compute_geodesic_distance, writer,
                             num_iters_max=splocs_max_itrs, num_admm_iterations=splocs_admm_num_itrs):

        R = residual_init
        # save("residual_init_componentClasses", R)
        C = []
        W = []
        s = None
        for k in range(self.numComp):
            # find the vertex explaining the most variance across the residual animation
            magnitude = (R ** 2).sum(axis=2)  # 3D frames have been collapsed to 1D frames containing L_2 norms
            idx = argmax(
                magnitude.sum(axis=0))  # collapse the (F, N) to only (N) then find the argmax over all vertices

            # find linear component explaining the motion of this vertex
            U, sing, Vt = svd(R[:, idx, :].reshape(R.shape[0], -1).T, full_matrices=False)
            # R[:,idx,:].reshape(R.shape[0], -1).T is the (3,F) tensor associated to the vertex==id
            wk = sing[0] * Vt[0, :]  # weights (F,1): only the one associated to first mode of the svd

            if self.support == 'local':
                # weight according to their projection onto the constraint set
                # this fixes problems with negative weights and non-negativity constraints
                wk_proj = self.project_weight(wk)
                wk_proj_negative = self.project_weight(-wk)
                wk = wk_proj \
                    if norm(wk_proj) > norm(wk_proj_negative) \
                    else wk_proj_negative
                s = 1 - self.compute_support_map(idx, snapshots_compute_geodesic_distance,
                                                 self.smooth_min_dist, self.smooth_max_dist)  # (N,)

            # solve for optimal component inside support map
            # wk is (F,), R is (F, N, 3), tensordot(wk, R, (0, 0)) is (N, 3), s[:,newaxis] is (N, 1)
            # components are normalized by ||wk||, each component is (N,3)
            if self.support == 'local':
                ck = (tensordot(wk, R, (0, 0)) * s[:, newaxis]) \
                     / inner(wk, wk)
            else:
                ck = tensordot(wk, R, (0, 0)) / inner(wk, wk)

            C.append(ck)
            W.append(wk)

            # update residual
            R -= outer(wk, ck).reshape(R.shape)

            singList = [k, sing[0]]

            if self.storeSingVal:
                writer.writerow(singList)

        self.comps = array(C)
        # save('C_classes_200', self.comps)
        self.weigs = array(W).T

        if self.basesType == 'SPLOCS':
            R_flat_init = R.copy()
            self.splocs_glob_optimization(num_iters_max, num_admm_iterations,
                                     R_flat_init, snapshots_compute_geodesic_distance)

    def splocs_glob_optimization(self, num_iters_max, num_admm_iterations,
                                 R, compute_geodesic_distance, sparsity_lambda=splocs_lambda, rho=splocs_rho):
        # prepare auxiluary variables
        Lambda = empty((self.numComp, self.pos_snapshots.nVerts))
        U = zeros((self.numComp, self.pos_snapshots.nVerts, 3))
        C = self.comps.copy()
        W = self.weigs.copy()
        X = self.pos_snapshots.snapTensor.copy()
        # main global optimization
        for it in range(num_iters_max):
            # update weights
            Rflat = R.reshape(self.pos_snapshots.frs, self.pos_snapshots.nVerts*3)
            for k in range(self.numComp):
                Ck = C[k].ravel()
                Ck_norm = inner(Ck, Ck)
                if Ck_norm <= 1.e-8:   # prevent divide by zero
                    # the component seems to be zero everywhere, so set it's activation to 0 also
                    W[:, k] = 0
                    continue

                # block coordinate descent update
                Rflat += outer(W[:, k], Ck)
                opt = dot(Rflat, Ck) / Ck_norm
                W[:, k] = self.project_weight(opt)
                Rflat -= outer(W[:, k], Ck)
            # update spatially varying regularization strength
            for k in range(self.numComp):
                ck = C[k]
                # find vertex with biggest displacement in component and compute support map around it
                idx = (ck**2).sum(axis=1).argmax()
                support_map = self.compute_support_map(idx, compute_geodesic_distance,
                                                  self.smooth_min_dist, self.smooth_max_dist)
                # update L1 regularization strength according to this support map
                Lambda[k] = sparsity_lambda * support_map

            # update components
            Z = C.copy() # dual variable
            # pre-factor linear solve in ADMM
            G = dot(W.T, W)
            c = dot(W.T, X.reshape(X.shape[0], -1))
            solve_prefactored = cho_factor(G + rho * eye(G.shape[0]))

            # ADMM iterations
            for admm_it in range(num_admm_iterations):
                C = cho_solve(solve_prefactored, c + rho * (Z - U).reshape(c.shape)).reshape(C.shape)
                Z = self.prox_l1l2(Lambda, C + U, 1. / rho)
                U = U + C - Z
            # set updated components to dual Z,
            # this was also suggested in [Boyd et al.] for optimization of sparsity-inducing norms
            C = Z
            # evaluate objective function
            R = X - tensordot(W, C, (1, 0))  # residual
            sparsity = sum(Lambda * sqrt((C**2).sum(axis=2)))
            E_rms = norm(R)/sqrt(3*self.pos_snapshots.nVerts*self.pos_snapshots.frs)   # Kavan et.al. 2010
            energy = (R**2).sum() + sparsity

            # TODO convergence check
            print("itr %03d, Energy =%f, Error =%f" % (it, energy, E_rms))

        print("Computed '", self.basesType, "' bases size ", self.comps.shape)

    def test_convergence(self, start, end, step):

        snapshots = self.pos_snapshots.snapTensor.copy()
        # snapshots_inf_norm = argmax(linalg.norm(snapshots.reshape(e, p, -1), ord='fro', axis=(1, 2)))
        denom = sqrt(3*self.pos_snapshots.nVerts *self.pos_snapshots.frs)

        for k in range(start, end + 1, step):
            res = snapshots[:, :, 0] - tensordot(self.weigs[:, :k], self.comps[:k, :, 0], axes=([1, 0]))  # residual
            E_rms = norm(res)/denom  # Kavan et.al. 2010
            print(k, E_rms)

    @staticmethod
    def prox_l1l2(Lambda, x, beta):
        xlen = sqrt((x**2).sum(axis=-1))
        with errstate(divide='ignore'):
            shrinkage = maximum(0.0, 1 - beta * Lambda / xlen)
        return x * shrinkage[..., newaxis]

    def compute_components_store_singvalues(self, store_bases_dir):
        residual_init = self.pos_snapshots.snapTensor.copy()
        compute_geodesic_distance = self.pos_snapshots.compute_geodesic_distance

        headerSing = ['component', 'singVal']

        file_name = os.path.join(store_bases_dir, "PCA_singValues")
        if self.storeSingVal:
            with open(file_name + '.csv', 'w', encoding='UTF8') as singFile:
                writer = csv.writer(singFile)
                writer.writerow(headerSing)

                self.extract_k_components(residual_init, compute_geodesic_distance, writer)
            singFile.close()
        else:
            self.extract_k_components(residual_init, compute_geodesic_distance, None)

    def post_process_components(self):

        if q_standarize:
            # undo scaling
            self.comps /= self.pos_snapshots.pre_scale_factor  # (K, N, 3)

            # undo the mean-subtraction (important for bases visualisation on the full character mesh)
            self.comps += self.pos_snapshots.mean[newaxis]  # (K, N, 3) + (1, N, 3)

        if q_orthogonal:
            # orthogonalize per dimension
            for l in range(self.comps.shape[2]):
                self.comps[:, :, l] = orth(self.comps[:, :, l].T).T

        if q_massWeight:
            # compute M^{-1/2} U for each dimension x, y, z.
            assert self.comps.shape[1] == self.pos_snapshots.invMassL.shape[0]
            self.comps *= self.pos_snapshots.invMassL[:, None]

        # components shifted by mean shape
        testSparsity(self.basesType + " bases", self.comps, 2)
        test_linear_indpendency(self.comps, 3, self.numComp)

        # # only were non zero weights # (K, N, 3)
        # if self.pos_snapshots.rest_shape == 'first':
        #     self.comps_nomean = self.comps - self.pos_snapshots.verts[0]
        #
        # elif self.pos_snapshots.rest_shape == 'average':
        #     self.comps_nomean = self.comps - mean(self.pos_snapshots.verts, axis=0)
        #
        # testSparsity("Mean-not-added " + self.basesType + " bases", self.comps_nomean
        #              , 2)
        # test_linear_indpendency(self.comps_nomean, 3, self.numComp)

        if q_orthogonal:
            self.is_utmu_orthogonal()  # test U^T M U = I (K x K)

        print("... Volkwein ("+str(q_massWeight)+")... standerized ("+str(q_standarize)+
              "), orthogonalized ("+str(q_orthogonal)+").")

    def is_utmu_orthogonal(self):
        print('... testing M orthogonality, U^T M U = I (K x K) ...', end='', flush=True)
        # comps = U^T
        for l in range(self.comps.shape[2]):
            Mu_l = self.comps[:, :, l].T * self.pos_snapshots.mass[:, None]   # M U
            utMu_l = dot(self.comps[:, :, l], Mu_l)  # U^T M U
            assert allclose(utMu_l, eye(self.comps.shape[0]))
            Mu_l, utMu_l = None, None
        print('(True).')

    # def get_storage_files_names(self, standarized, massWeighted, orthogonalized, supported, testingComputations):
    #     """
    #     Form the name of the storing files automatically depending on the given bases type and its characteristics
    #     """
    #     fileNameBases = 'q' + str(self.basesType)
    #     fileNamenoMeanBases = 'qnoMean' + str(self.basesType)
    #     file_name_sing = 'singVals_q' + str(self.basesType)
    #
    #     fileNameBases += massWeighted + standarized + supported + orthogonalized + testingComputations
    #     fileNamenoMeanBases += massWeighted + standarized + supported + orthogonalized + testingComputations
    #     file_name_sing += massWeighted + standarized + supported + orthogonalized + testingComputations
    #     self.fileNameBases = fileNameBases
    #     self.fileNamenoMeanBases = fileNamenoMeanBases
    #     self.file_name_sing = file_name_sing

    def store_components_to_files(self, output_bases_dir, start, end, step, fileType):
        """
        fileType can be either '.bin' or '.npy'
        """
        print('Storing bases ...', end='', flush=True)
        numframes, numverts = self.pos_snapshots.frs, self.pos_snapshots.nVerts

        basesFile = os.path.join(output_bases_dir, self.fileNameBases)
        # basesnoMeanFile = os.path.join(output_bases_dir, self.fileNamenoMeanBases)
        # store separate .bin for different numbers of components
        for k in range(start, end + 1, step):
            store_components(basesFile, numframes, k, numverts, 3, self.comps[:k, :, :], fileType)
            # store_components(basesnoMeanFile, numframes, k, numverts, 3, self.comps_nomean[:k, :, :], fileType)
        print('done.')

    def store_animations(self, output_bases_dir):

        output_components = os.path.join(output_bases_dir, self.output_components_file)
        output_animation = os.path.join(output_bases_dir, self.output_animation_file)

        # save components as animation
        with h5py.File(output_components, 'w') as f:
            f['default'] = self.pos_snapshots.verts[0]
            f['tris'] = self.pos_snapshots.tris
            for i, c in enumerate(self.comps):
                f['comp%03d' % i] = c
        f.close()
        # save encoded animation including the weights
        if self.output_animation_file:
            with h5py.File(output_animation, 'w') as wh5:
                # wh5['verts'] = tensordot(self.weigs, self.comps, (1, 0))
                wh5['verts'] = self.comps
                wh5['tris'] = self.pos_snapshots.tris
                wh5['weights'] = self.weigs
            wh5.close()

