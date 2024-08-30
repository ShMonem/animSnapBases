# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import numpy as np
import struct
from numpy.linalg import matrix_rank
from numpy import save, count_nonzero


def store_components(fileName, F, K, N, dim, basesTensor, extension='.bin',  colName='K'):
    """
    :param fileName: file to store data
    :param F: num of frames/snapshots used to compute the bases
    :param K: num of bases, in the nonlinearFun bases case 'Kp' is expected instead of 'K'
    :param N: either 'nVerts' for positions bases, or 'ep' for nonlinear case
    :param dim: =3 (x, y, z)
    :param basesTensor: of size (K, N, 3) or (Kp, ep, 3)
    :param extension: file type
    :return: stored file in the given path
    """
    assert basesTensor.shape == (K, N, dim)
    if extension == '.bin':
        with open(fileName + '_F' + str(F) + colName + str(K) + extension, 'wb') as doc0:
            doc0.write(struct.pack("<i", N))  # write a 4 byte integer in little endian
            doc0.write(struct.pack("<i", dim * K))  # write a 4 byte integer in little endian
            for d in range(dim):
                for k in range(K):
                    for i in range(N):
                        value = basesTensor[k, i, d]
                        doc0.write(struct.pack("<d", value))  # write a double precision (8 byte) in little endian
        doc0.close()

    if extension == '.npy':  # important in case we want to compare parts of the stored components as matrices
        np.save(fileName + str(F) + 'K' + str(K), basesTensor)

def testSparsity(mat):
    """
    :param name: bases type
    :param mat: expects tensor of size (:, :, 3)
    :param test_dim: always the third dim (x, y, z)
    :return: informs if the mat slices along test dimension are sparse
    """
    assert mat.shape[2] == 3
    # print('    ' + name + '...', end='', flush=True)
    sparPerList = []
    for l in range(3):
        sparPer = 1 - (np.count_nonzero(mat[:, :, l]) / mat[:, :, l].size)
        sparPerList.append(sparPer)
    if min(sparPerList) > 0.5:
        print("sparse, min %" + str(100*min(sparPerList)) + " zero entries.")
    else:
        print("not sparse.")


def test_linear_dependency(mat, test_dim_range, expected_rank):
    """
        :param mat: expects tensor of size (:, :, 3)
        :param test_dim_range: typically for 3 (x,y,z)
        :param expected_rank: 'K' in case positions bases, and 'Kp' in case nonlinearFun bases
        :return: informs if the mat slices along test dimension are not linear independent
        """
    assert mat.shape[2] == 3

    for j in range(test_dim_range):
        try:
            matrix_rank(mat[:, :, j]) == expected_rank
        except:
            print(str(mat) + "is not linear independent, with rank: " + str(matrix_rank(mat[:, :, j]))
                  + " != " + str(expected_rank) + ".")


def store_vector(fileName, F, K, points, extension='.bin', colName='K'):
    """
    :param fileName: file to store data
    :param F: num of frames/snapshots used to compute the bases
    :param K: num of bases blocks in the nonlinearFun bases
    :param points: array contains the indices at which the nonlinearFunc bases are computed
    :param extension: files type
    :return: stored file in the given path
    """
    assert points.shape[0] == K
    if extension == '.bin':
        with open(fileName + '_F' + str(F) + colName + str(K) + extension, 'wb') as doc0:
            doc0.write(struct.pack("<i", K))  # write a 4 byte integer in little endian
            doc0.write(struct.pack("<i", 1))  # write a 4 byte integer in little endian
            for k in range(K):
                value = points[k]
                doc0.write(struct.pack("<d", value))  # write a double precision (8 byte) in little endian
        doc0.close()

    if extension == '.npy':  # important in case we want to compare parts of the stored components as matrices
        np.save(fileName + str(F) + 'K' + str(K), points)

