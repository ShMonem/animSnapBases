# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import os
import csv
import h5py
import numpy as np
from numpy import maximum, argmax, clip, save, tensordot, inner, outer, array, newaxis, empty, zeros, dot, eye, sum,\
                  sqrt, errstate, maximum, allclose, mean
from scipy.linalg import svd, norm, cho_factor, cho_solve, cholesky, orth
from utils.utils import store_components, testSparsity, test_linear_dependency
from snapbases.posSnapshots import posSnapshots
from config.config import Config_parameters
from utils.utils import log_time


class posComponents:  # Components == bases
    def __init__(self, param: Config_parameters):
        # bases type, can only be 'PCA' and 'SPLOCS'
        self.basesType = param.vertPos_bases_type
        assert self.basesType == 'PCA' or self.basesType == 'SPLOCS'

        # position snapshots from which the components are computed are loaded and pre-processed if required
        training_aligned_snapshots_h5_file = os.path.join(param.aligned_snapshots_directory, param.train_aligned_snapshots_animation_file)
        testing_aligned_snapshots_h5_file = os.path.join(param.aligned_snapshots_directory, param.test_aligned_snapshots_animation_file)

        self.pos_snapshots = posSnapshots(training_aligned_snapshots_h5_file, testing_aligned_snapshots_h5_file,
                                          param.vertPos_rest_shape, param.vertPos_masses_file,
                                          param.tet_mesh_file,
                                          param.q_standarize, param.q_massWeight)

        self.numComp = param.vertPos_numComponents  # max number of components
        self.support = param.q_support  # can be 'local' or 'global'

        self.storeSingVal = param.store_vertPos_PCA_sing_val  # boolean

        self.comps = None  # bases tensor : U + average shape
        self.weigs = None  # weights matrix
        self.ortho_comps = None  # orthogonal bases tensor
        self.smooth_min_dist = param.vertPos_smooth_min_dist  # minimum geodesic distance for support map, d_min_in paper
        self.smooth_max_dist = param.vertPos_smooth_max_dist
        self.output_components_file = "components.h5"
        # self.output_animation_file = "animations.h5"

        self.measures_at_largeDeforVerts = None  # singVals & resNorm computed at 'K' largest deformation verts
        self.fileNameBases = "q_pos_"

        self.param = param

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

    @log_time("")
    def extract_k_components(self, writer,
                             num_iters_max=20, num_admm_iterations=10):
        R = self.pos_snapshots.snapTensor.copy()
        snapshots_compute_geodesic_distance = self.pos_snapshots.compute_geodesic_distance
        # save("residual_init_componentClasses", R)
        C = []
        W = []
        s = None
        self.measures_at_largeDeforVerts = []
        for k in range(self.numComp):
            # find the vertex explaining the most variance across the residual animation
            magnitude = (R ** 2).sum(axis=2)  # 3D frames have been collapsed to 1D frames containing L_2 norms
            idx = argmax(
                magnitude.sum(axis=0))  # collapse the (F, N) to only (N) then find the argmax over all vertices

            # find linear component explaining the motion of this vertex
            _, sing, Vt = svd(R[:, idx, :].reshape(R.shape[0], -1).T, full_matrices=False)
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

            singList = [k, sing[0], norm(R)]
            self.measures_at_largeDeforVerts.append(singList)

            if self.storeSingVal:
                writer.writerow(singList)

        self.comps = array(C)
        # save('C_classes_200', self.comps)
        self.weigs = array(W).T
        self.measures_at_largeDeforVerts = array(self.measures_at_largeDeforVerts)

        if self.basesType == 'SPLOCS':
            R_flat_init = R.copy()
            self.splocs_glob_optimization(self.param.splocs_max_itrs, self.param.splocs_admm_num_itrs,
                                     R_flat_init, snapshots_compute_geodesic_distance)

        print("Computed '", self.basesType, "' bases size ", self.comps.shape)

    @log_time("")
    def splocs_glob_optimization(self, num_iters_max, num_admm_iterations,
                                 R, compute_geodesic_distance):
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
                    # the component seems to be zero everywhere, so set its activation to 0 also
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
                Lambda[k] = self.param.splocs_lambda * support_map

            # update components
            Z = C.copy() # dual variable
            # pre-factor linear solve in ADMM
            G = dot(W.T, W)
            c = dot(W.T, X.reshape(X.shape[0], -1))
            solve_prefactored = cho_factor(G + self.param.splocs_rho * eye(G.shape[0]))

            # ADMM iterations
            for admm_it in range(num_admm_iterations):
                C = cho_solve(solve_prefactored, c + self.param.splocs_rho * (Z - U).reshape(c.shape)).reshape(C.shape)
                Z = self.prox_l1l2(Lambda, C + U, 1. / self.param.splocs_rho)
                U = U + C - Z
            # set updated components to dual Z,
            # this was also suggested in [Boyd et al.] for optimization of sparsity-inducing norms
            C = Z
            # evaluate objective function
            R = X - tensordot(W, C, (1, 0))  # residual
            sparsity = sum(Lambda * sqrt((C**2).sum(axis=2)))
            E_rms = norm(R)/sqrt(3*self.pos_snapshots.nVerts*self.pos_snapshots.frs)   # Kavan et.al. 2010
            energy = (R**2).sum() + sparsity

            # TODO convergence check plot
            print("itr %03d, Energy =%f, Error =%f" % (it, energy, E_rms))

    @log_time("")
    def test_convergence(self, start, end, step, writer=None):
        snapshots = self.pos_snapshots.snapTensor.copy()   # (F, N, 3)
        # snapshots_inf_norm = argmax(linalg.norm(snapshots.reshape(e, p, -1), ord='fro', axis=(1, 2)))
        denom = sqrt(3*self.pos_snapshots.nVerts *self.pos_snapshots.frs)

        fro_err =[]
        rel_err_x = []
        rel_err_y = []
        rel_err_z = []
        max_err = []
        for k in range(start, end + 1, step):
            reconstructed  = zeros(snapshots.shape)
            for f in range(snapshots.shape[0]):
                reconstructed[f, :, :] =  tensordot(self.weigs[f, :k][newaxis, :] , self.comps[:k, :, :], axes=([1,0]))

            fro_err.append(self.frobenius_error(snapshots, reconstructed))
            rel_err = self.relative_error_per_component(snapshots, reconstructed)
            rel_err_x.append(rel_err[0])
            rel_err_y.append(rel_err[1])
            rel_err_z.append(rel_err[2])
            max_err.append(self.max_pointwise_error(snapshots, reconstructed))

        return fro_err, max_err, rel_err_x, rel_err_y, rel_err_z


    @staticmethod
    def frobenius_error(f, f_reconstructed):
        """
        Computes the Frobenius norm of the reconstruction error.
        """
        error = f - f_reconstructed
        return  norm(error)

    @staticmethod
    def relative_error_per_component(f, f_reconstructed):
        """
        Computes the relative error for each component.
        """
        relative_errors = []  # Relative errors for (x, y, z)

        for i in range(3):
            norm_original =  norm(f[:, :, i])
            norm_error =  norm(f[:, :, i] - f_reconstructed[:, :, i])
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
        error =  np.abs(f - f_reconstructed)
        val = np.max(error)/ np.max(f)
        return  val

    @staticmethod
    def prox_l1l2(Lambda, x, beta):
        xlen = sqrt((x**2).sum(axis=-1))
        with errstate(divide='ignore'):
            shrinkage = maximum(0.0, 1 - beta * Lambda / xlen)
        return x * shrinkage[..., newaxis]

    @log_time("")
    def compute_components_store_singvalues(self):

        headerSing = ['component', 'singVal', 'norm_R']

        file_name = os.path.join(self.param.vertPos_output_directory, self.param.name+"_posBases_pcaExtraction_singValues_errorNorm")
        if self.storeSingVal:
            with open(file_name + '.csv', 'w', encoding='UTF8') as singFile:
                writer = csv.writer(singFile)
                writer.writerow(headerSing)

                self.extract_k_components(writer )
            singFile.close()
        else:
            self.extract_k_components(None)

    @log_time("")
    def post_process_components(self):
        print("Post-processing pos components ...")
        if self.param.q_standarize:
            # undo scaling
            self.comps /= self.pos_snapshots.pre_scale_factor  # (K, N, 3)

            # undo the mean-subtraction (important for bases visualisation on the full character mesh)
            self.comps += self.pos_snapshots.mean[newaxis]  # (K, N, 3) + (1, N, 3)

        if self.param.q_orthogonal:
            # orthogonalize per dimension
            for l in range(self.comps.shape[2]):
                self.comps[:, :, l] = orth(self.comps[:, :, l].T).T

        if self.param.q_massWeight:
            # compute M^{-1/2} U for each dimension x, y, z.
            assert self.comps.shape[1] == self.pos_snapshots.invMassL.shape[0]
            self.comps *= self.pos_snapshots.invMassL[:, None]

        # few tests
        testSparsity(self.comps)
        test_linear_dependency(self.comps, 3, self.numComp)

        if self.param.q_orthogonal:
            self.is_utmu_orthogonal()  # test U^T M U = I (K x K)

        print("... Volkwein ("+str(self.param.q_massWeight)+")... standerized ("+str(self.param.q_standarize) +
              ")... support (" +str(self.support) + "), orthogonalized ("+str(self.param.q_orthogonal)+").")

    @log_time("")
    def is_utmu_orthogonal(self):
        print('... testing M orthogonality, U^T M U = I (K x K) ...', end='', flush=True)
        # comps size (K , N, 3)
        for l in range(self.comps.shape[2]):
            Mu_l = self.comps[:, :, l].T * self.pos_snapshots.mass[:, None]   # M U
            utMu_l = dot(self.comps[:, :, l], Mu_l)  # U^T M U
            assert allclose(utMu_l, eye(self.comps.shape[0]))
            Mu_l, utMu_l = None, None
        print('(True).')

    @log_time("")
    def store_components_to_files(self, start, end, step, fileType):
        """
        fileType can be either '.bin' or '.npy'
        """
        print('Storing bases ...', end='', flush=True)
        numframes, numverts = self.pos_snapshots.frs, self.pos_snapshots.nVerts

        basesFile = os.path.join(self.param.vertPos_output_directory, self.fileNameBases)
        # store separate .bin for different numbers of components
        for k in range(start, end + 1, step):
            store_components(basesFile, numframes, k, numverts, 3, self.comps[:k, :, :], fileType, 'K')
        print('done.')

    @log_time("")
    def store_animations(self, output_bases_dir):

        output_components = os.path.join(output_bases_dir, self.output_components_file)
        # output_animation = os.path.join(output_bases_dir, self.output_animation_file)

        # save components as animation
        with h5py.File(output_components, 'w') as f:
            f['default'] = self.pos_snapshots.verts[0]
            f['tris'] = self.pos_snapshots.tris
            for i, c in enumerate(self.comps):
                f['comp%03d' % i] = c
        f.close()

    @log_time("")
    def test_basesSingVals(self):
        """
        Computes normalized singular values along all Kp vectors on the already fully-extracted PCA bases
        :return:
        s = [sx, sy, sz] normalized for each dim separately
        """
        bases = self.comps.copy()  # (K, N, 3)
        s = empty((bases.shape[0], 3))
        for i in range(3):
            U, sing, Vt = svd(bases[:, :, i], full_matrices=False)
            s[:, i] = sing / sing.max()
        # print('min sing values over dimensions:', s[:, 0].min(), s[:, 1].min(), s[:, 2].min())
        return s
