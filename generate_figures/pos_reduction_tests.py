import numpy as np
import matplotlib.pyplot as plt
from utils.utils import log_time
import csv
from utils.utils import testSparsity, test_linear_dependency
import os
from snapbases.posComponents import posComponents

from config.config import Config_parameters

@log_time("")
def tets_plots_pca(bases: posComponents, param:Config_parameters):
    """
    Plots different reconstruction errors for varying reduction dimensions "r".
    :param f: Original tensor (T, N, 3)
    :param V_f: Basis tensor (N, max_r, 3)
    :param max_r: Maximum reduction dimension (number of modes)
    """
    vertPos_output_directory = param.vertPos_output_directory

    def ren_tests(bases: posComponents, param:Config_parameters):
        k = bases.numComp
        r_values = range(1, k + 1)

        # PCA tests --------------------------------------------------------------------------------------------------------
        plt.figure('Error measures for PCA - pos bases.', figsize=(20, 10))

        store_kp_singVals = True

        rows = 1
        cols = 3
        plt.subplot(rows, cols, 1)

        # singular values at 'K' largest deformation blocks during PCA bases extarction
        # singVals starts from entry 2 in the measures_at_largeDeforVerts array
        mark = ['bo', 'ro', 'go']

        plt.plot(r_values, bases.measures_at_largeDeforVerts[:, 1] /
                 bases.measures_at_largeDeforVerts[:, 1].max(), mark[1], ls='-.', label=f'$\sigma$')

        plt.legend(loc='upper center')
        plt.xlabel('Reduction Dimension (r)')
        plt.ylabel('Normalized $\sigma$')
        plt.title("singVals at larg_deformation verts")
        # plt.yscale("log")
        # plt.xticks(np.arange(1, k+1, 1))
        plt.legend()

        plt.subplot(rows, cols, 2)
        # residual_norm values at 'K' largest deformation blocks: norm(R) is expected to be the 3rd entry
        plt.plot(r_values, bases.measures_at_largeDeforVerts[:, 2], 'rv', ls='-', label='$\| R_{pca} \|_F$ blocks')
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
            header_ = ['row','x', 'y', 'z']
            file_name_ = os.path.join(vertPos_output_directory, param.name + "_posBases_xyz_fullBasesRange_K_singVals")
            with open(file_name_ + '.csv', 'w', encoding='UTF8') as dataFile_:
                writer_ = csv.writer(dataFile_)
                writer_.writerow(header_)

                s = bases.test_basesSingVals()
                for row in range(s.shape[0]):
                    writer_.writerow([row +1 ,s[row, 0], s[row, 1], s[row, 2]])

            dataFile_.close()
        else:
            s = bases.test_basesSingVals()

        plt.plot(r_values, s[:, 0], 'bo', ls='--', label='$\sigma_x$')
        plt.plot(r_values, s[:, 1], 'ro', ls='--', label='$\sigma_y$')
        plt.plot(r_values, s[:, 2], 'go', ls='--', label='$\sigma_z$')
        plt.legend(loc='upper center')
        plt.xlabel('Reduction Dimension (r)')
        plt.ylabel('Fro. nom')
        plt.title("Normalized singVal(bases), full K range")
        # plt.yscale("log")
        # plt.xticks(values)
        plt.legend()
        fig_name = os.path.join(vertPos_output_directory, 'pca_extraction_tests')
        plt.savefig(fig_name)
        # End of PCA tests -------------------------------------------------------------------------------------------------

        # After post-process tests
        testSparsity(bases.comps)
        test_linear_dependency(bases.comps, 3, bases.numComp)

        if param.q_orthogonal:
            bases.is_utmu_orthogonal()  # test U^T M U = I (K x K)

        # Convergence tests with rotations ---------------------------------------------------------------------------------

    ren_tests(bases, param)


# Convergence tests ------------------------------------------------------------------------------------------------
        # Note: This convergences tests section is not relaible
        # frobenius_errors, max_errors, relative_errors_x, relative_errors_y, relative_errors_z = bases.test_convergence(1, k, 1)
        #
        # # Plot Frobenius and inf norm error
        # plt.figure('Error measures for PCA pos bases', figsize=(20, 10))
        #
        # plt.subplot(1, 2, 1)
        # plt.plot(frobenius_errors, label='Frobenius Error', marker='o')
        # plt.plot(r_values, max_errors, label='Inf Error', marker='o')
        # plt.xlabel('Reduction Dimension (r)')
        # plt.ylabel('Error')
        # plt.title('Frobenius Norm')
        # plt.yscale("log")
        # # Set x-ticks to integers only
        # # plt.xticks(np.arange(1, k + 1, 1)) # range 0 <= r <= numComponents
        # plt.legend()
        #
        # # Plot Relative Errors for each component (x, y, z)
        # plt.subplot(1, 2, 2)
        # relative_error = np.sum(np.array([relative_errors_x, relative_errors_y, relative_errors_z]), axis=0)
        # plt.plot(r_values, relative_errors_x, label='Relative Error X', marker='o')
        # plt.plot(r_values, relative_errors_y, label='Relative Error Y', marker='x')
        # plt.plot(r_values, relative_errors_z, label='Relative Error Z', marker='s')
        # plt.plot(r_values, relative_error, label='sumRelative Error', marker='v')
        # plt.xlabel('Reduction Dimension (r)')
        # plt.ylabel('Relative Error')
        # plt.title('Relative Errors per Component (X, Y, Z)')
        # plt.yscale("log")
        # # plt.xticks(np.arange(1, k+1, 1))
        # plt.legend()
        #
        # # plt.tight_layout()
        # fig_name = os.path.join(vertPos_output_directory, 'deim_convergence_tests')
        # plt.savefig(fig_name)