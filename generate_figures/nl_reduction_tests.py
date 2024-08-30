import numpy as np
import matplotlib.pyplot as plt
from config.config import constProj_standarize, constProj_massWeight, constProj_orthogonal, constProj_p_size,\
                            constProj_output_directory
from utils.utils import testSparsity, test_linear_dependency
import os
from snapbases.constraintsComponents import constraintsComponents


def run_pca_bases_constriants_tests(nlConst_bases: constraintsComponents, test_pca = True, test_deim= True):
    """
    :param nlConst_bases: type constraintsComponents class
    :return: few tests and visualizations of some results
    """
    if test_pca:
        # components shifted by mean shape
        print('Testing ' + nlConst_bases.basesType + " Fully post-processed bases ...", end='', flush=True)
        testSparsity(nlConst_bases.comps)
        test_linear_dependency(nlConst_bases.comps, 2, constProj_p_size * nlConst_bases.numComp)

        if constProj_orthogonal:
            nlConst_bases.is_utmu_orthogonal()  # test U^T M U = I (K x K)

        print("Bases computed: Volkwein ("+str(constProj_massWeight)+")... standerized ("+str(constProj_standarize) +
              "), orthogonalized ("+str(constProj_orthogonal)+").")

        plt.figure('Error measures for PCA bases', figsize=(20, 10))
        rows = 2
        cols = 2
        plt.subplot(rows, cols, 1)
        plt.title("Normalized singVal(R) at larg_defor. K blocks")
        # singular values at 'K' largest deformation blocks during PCA bases extarction
        plt.plot(nlConst_bases.measures_at_largeDeforVerts[:, 2]/nlConst_bases.measures_at_largeDeforVerts[:, 2].max(),
                 'bo', ls='-.', label='$\sigma_x$')
        plt.plot(nlConst_bases.measures_at_largeDeforVerts[:, 3]/nlConst_bases.measures_at_largeDeforVerts[:, 3].max(),
                 'ro', ls='-.', label='$\sigma_y$')
        plt.plot(nlConst_bases.measures_at_largeDeforVerts[:, 4]/nlConst_bases.measures_at_largeDeforVerts[:, 4].max(),
                 'go', ls='-.', label='$\sigma_z$')
        plt.legend(loc='upper center')

        plt.subplot(rows, cols, cols+1)
        plt.title("Normalized singVal(bases), full Kp range")
        # singular vals for the bases over full Kp range
        s = nlConst_bases.test_basesSingVals()
        plt.plot(s[:, 0], 'bo', label='$\sigma_x$')
        plt.plot(s[:, 1], 'ro', label='$\sigma_y$')
        plt.plot(s[:, 2], 'go', label='$\sigma_z$')
        plt.legend(loc='upper center')

        plt.subplot(rows, cols, 2)
        plt.title("norm(R) at K PCA-bases-extraction")
        # residual_norm values at 'K' largest deformation blocks
        plt.plot(nlConst_bases.measures_at_largeDeforVerts[:, 5],
                 'rv', ls='-.', label='$\| R_{pca} \|_F$ blocks')
        plt.legend(loc='upper center')


        error_kp_w = []  # reconstruction error at diff. 'K' PCA bases-blocks only at large deformation |S^T WC - S^T p(t)|
        error_full_w = []  # reconstruction error at different 'K' PCA bases-blocks full dim |WC - p(t)|
        nlConst_bases.test_PCA_reconstruction_error_at_kp_bases_using_weights(error_full_w, error_kp_w)

        plt.subplot(rows, cols, cols + 2)
        plt.title("re-construction error PCA")
        plt.plot(np.array(error_kp_w), 'g+', ls='-.', label='$\|S^T WC - S^T p(t)\|_F$')
        plt.plot(np.array(error_full_w), 'r+', ls='-.', label='$\|WC - p(t)\|_F$')
        plt.legend(loc='upper center')

        fig_name = os.path.join(constProj_output_directory,  'pca_tests')
        plt.savefig(fig_name)

    if test_deim:

        plt.figure('Error measures for DEIM ', figsize=(20, 10))
        rows = 2
        cols = 2

        error_full = []  # reconstruction error at different 'K' PCA bases-blocks full dim |Vc(t) - p(t)|
        error_kp = []  # reconstruction error at diff. 'K' PCA bases-blocks only at large deformation |S^T Vc(t) - S^T p(t)|
        nlConst_bases.test_DEIM_reconstruction_error_at_different_kp_bases(error_full, error_kp)

        plt.subplot(rows, cols, 1)
        plt.title("re-construction error DEIM")
        plt.plot(np.array(error_kp), 'g+', ls='-.', label='$\|S^T Vc(t) - S^T p(t)\|$')
        plt.plot(np.array(error_full), 'r+', ls='-.', label='$\|Vc(t) - p(t)\|$')
        plt.legend(loc='upper center')

        plt.subplot(rows, cols, 2)
        plt.title("projection error DEIM")
        plt.plot(nlConst_bases.deim_res_norm, 'g+', ls='-.', label='$\|V v_k - c \|$')
        plt.legend(loc='upper center')

        fig_name = os.path.join(constProj_output_directory, 'deim_tests')
        plt.savefig(fig_name)

    # plt.show()