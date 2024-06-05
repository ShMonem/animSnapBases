import struct
from numpy.linalg import matrix_rank
from numpy import save, count_nonzero


def store_components(fileName, F, K, N, dim, basisPlain, extension='.bin'):
    if extension == '.bin':
        with open(fileName + str(F) + 'K' + str(K) + extension, 'wb') as doc0:
            doc0.write(struct.pack("<i", N))  # write a 4 byte integer in little endian
            doc0.write(struct.pack("<i", 3 * K))  # write a 4 byte integer in little endian
            for d in range(dim):
                for k in range(K):
                    for i in range(N):
                        value = basisPlain[k, i, d]
                        doc0.write(struct.pack("<d", value))  # write a double precision (8 byte) in little endian
        doc0.close()

    if extension == '.npy':  # important in case we want to compare parts of the stored components as matrices
        save(fileName + str(F) + 'K' + str(K), basisPlain)


def testSparsity(name, mat, test_dim):
    print("... testing the sparsity of " + name, end='', flush=True)
    sparPerList = []
    for l in range(mat.shape[test_dim]):
        sparPer = 1 - (count_nonzero(mat[:, :, l]) / mat[:, :, l].size)
        sparPerList.append(sparPer)
    if max(sparPerList) > 0.5:
        print(" ... sparse, with min %" + str(100*min(sparPerList)) + " zero entries in each dimension")
    else:
        print(" ... not sparse.")


def test_linear_indpendency(mat, test_dim_range, expected_rank):
    for j in range(test_dim_range):
        try:
            matrix_rank(mat[:, :, j]) == expected_rank
        except:
            print(str(mat) + "is not linear independent, with rank: " + str(matrix_rank(mat[:, :, j]))
                  + " while " + str(expected_rank) + " was expected ")

