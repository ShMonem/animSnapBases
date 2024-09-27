import numpy as np
import matplotlib.pyplot as plt
from utils.utils import log_time

import csv

from config.config import constProj_orthogonal, constProj_output_directory
from utils.utils import testSparsity, test_linear_dependency
import os
from snapbases.constraintsComponents import constraintsComponents


@log_time(constProj_output_directory)
def plot_deim_reconstruction_errors(nlConst_bases: constraintsComponents, writer=None):
    """
    Plots different reconstruction errors for varying reduction dimensions "r".
    :param f: Original tensor (T, N, 3)
    :param V_f: Basis tensor (N, max_r, 3)
    :param max_r: Maximum reduction dimension (number of modes)
    """
    k = nlConst_bases.numComp
    p = nlConst_bases.nonlinearSnapshots.constraintsSize
    rp_values = range(p, k * p + 1, p)
    r_values = range(1, k + 1)

    # PCA tests --------------------------------------------------------------------------------------------------------
    plt.figure('Error measures for PCA.', figsize=(20, 10))

    store_kp_singVals = True

    rows = 1
    cols = 3
    plt.subplot(rows, cols, 1)

    # singular values at 'K' largest deformation blocks during PCA bases extarction
    plt.plot(r_values, nlConst_bases.measures_at_largeDeforVerts[:, 2] /
             nlConst_bases.measures_at_largeDeforVerts[:, 2].max(), 'bo', ls='-.', label='$\sigma_x$')

    plt.plot(r_values, nlConst_bases.measures_at_largeDeforVerts[:, 3] /
             nlConst_bases.measures_at_largeDeforVerts[:, 3].max(), 'ro', ls='-.', label='$\sigma_y$')

    plt.plot(r_values, nlConst_bases.measures_at_largeDeforVerts[:, 4] /
             nlConst_bases.measures_at_largeDeforVerts[:, 4].max(), 'go', ls='-.', label='$\sigma_z$')

    plt.legend(loc='upper center')
    plt.xlabel('Reduction Dimension (r)')
    plt.ylabel('Normalized $\sigma$')
    plt.title("singVals at larg_deformation points")
    # plt.yscale("log")
    plt.xticks(np.arange(1, k+1, 1))
    plt.legend()

    plt.subplot(rows, cols, 2)

    # residual_norm values at 'K' largest deformation blocks
    plt.plot(r_values, nlConst_bases.measures_at_largeDeforVerts[:, 5], 'rv', ls='-', label='$\| R_{pca} \|_F$ blocks')
    plt.legend(loc='upper center')
    plt.xlabel('Reduction Dimension (r)')
    plt.ylabel('Fro. nom')
    plt.title("norm(R) at K PCA bases extraction")
    # plt.yscale("log")
    plt.xticks(np.arange(1, k + 1, 1))
    plt.legend()

    plt.subplot(rows, cols, 3)
    # singular vals for the bases over full Kp range
    if store_kp_singVals:
        header_ = ['x', 'y', 'z']
        file_name_ = os.path.join(constProj_output_directory, "deim_Kp_singVals")
        with open(file_name_ + '.csv', 'w', encoding='UTF8') as dataFile_:
            writer_ = csv.writer(dataFile_)
            writer_.writerow(header_)

            s = nlConst_bases.test_basesSingVals(writer_)

        dataFile_.close()
    else:
        s = nlConst_bases.test_basesSingVals()

    values = range(1, k*p + 1, 1)
    plt.plot(values, s[:, 0], 'bo', ls='--', label='$\sigma_x$')
    plt.plot(values, s[:, 1], 'ro', ls='--', label='$\sigma_y$')
    plt.plot(values, s[:, 2], 'go', ls='--', label='$\sigma_z$')
    plt.legend(loc='upper center')
    plt.xlabel('Reduction Dimension (r)')
    plt.ylabel('Fro. nom')
    plt.title("Normalized singVal(bases), full Kp range")
    # plt.yscale("log")
    plt.xticks(values)
    plt.legend()
    fig_name = os.path.join(constProj_output_directory, 'pca_extraction_tests')
    plt.savefig(fig_name)
    # End of PCA tests -------------------------------------------------------------------------------------------------

    # After post-process tests
    testSparsity(nlConst_bases.comps)
    test_linear_dependency(nlConst_bases.comps, 3,
                           nlConst_bases.numComp * nlConst_bases.nonlinearSnapshots.constraintsSize)

    if constProj_orthogonal:
        nlConst_bases.is_utmu_orthogonal()  # test U^T M U = I (Kp x Kp)

    # DEIM tests -------------------------------------------------------------------------------------------------------

    frobenius_errors = []
    max_errors = []
    relative_errors_x = []
    relative_errors_y = []
    relative_errors_z = []

    f = nlConst_bases.nonlinearSnapshots.snapTensor
    for rp in rp_values:
        # Reconstruct the tensor for the current r
        f_reconstructed = nlConst_bases.deim_constructed(rp)

        # Compute various errors
        fro_error = nlConst_bases.frobenius_error(f, f_reconstructed)
        max_err = nlConst_bases.max_pointwise_error(f, f_reconstructed)
        rel_errors = nlConst_bases.relative_error_per_component(f, f_reconstructed)

        # Store errors
        frobenius_errors.append(fro_error)
        max_errors.append(max_err)
        relative_errors_x.append(rel_errors[0])
        relative_errors_y.append(rel_errors[1])
        relative_errors_z.append(rel_errors[2])

        if writer is not None:
            writer.writerow([rp//p, fro_error, max_err, rel_errors[0], rel_errors[1], rel_errors[2]])

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
    plt.xticks(np.arange(1, k + 1, 1)) # range 0 <= r <= numComponents
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
    plt.xticks(np.arange(1, k+1, 1))
    plt.legend()

    #plt.tight_layout()
    fig_name = os.path.join(constProj_output_directory, 'deim_convergence_tests')
    plt.savefig(fig_name)
    # End of DEIM tests ------------------------------------------------------------------------------------------------

    plt.show()