import os
import numpy as np
import struct
import sys
import cProfile

from config.config import constProj_input_snapshots_pattern, constProj_rest_shape, constProj_dim, \
                            constProj_masses_file, constProj_numFrames, constProj_p_size, constProj_massWeight,\
                            constProj_standarize


root_folder = os.getcwd()
profiler = cProfile.Profile()


class nonlinear_snapshots:
    """
    Constraints snapshots class
    read and pre-process a .bin input pre-recorded 'F' snapshots, each of size (ep, 3),
    if required the snapshots will be mass-weighted and/or standardized.
    """

    def __init__(self):

        self.snapshots_file = constProj_input_snapshots_pattern  # contains pre-aligned (only centered) snapshots
        self.rest_shape = constProj_rest_shape  # which frame to use as rest-shape ("first" or "average")
        self.dim = constProj_dim
        self.mass_file = constProj_masses_file  # file contains mass weights as one vector
        #self.verts = None  # vertices
        #self.tris = None  # faces
        self.frs = constProj_numFrames  # no. frames: F
        # self.nVerts = 0  # no. vertices: N
        self.constraintsSize = constProj_p_size  # 'p' in the paper. no. rows in each projection mat (=3, for TetStrain constraint)
        self.constraintVerts = 0  # numConstraints/'e' in the paper. no. verts (tetVerts) influenced by the constraints

        self.mean = None  # (nVerts, 3)
        self.pre_scale_factor = 1  # normalization factor

        #self.mass = None   # vertices masses vector
        self.massL = None  # Cholesky factorization L of mass matrix Mass = L^T L
        self.invMassL = None  # Choesky factorisation inverse L^{-1}

        self.snapTensor = None  # preprocessed snapshots tensors on which we compute components (basis/ modes)
                                # expected size of (F, ep, 3)


        # One time call: compute snapTensor
        self.snapshots_precomputations()

    def snapshots_precomputations(self):
        """
        One time snapshots loading and possibly pre-processing. Options are:
        standarize (note: this step includes also geodesics distances computation),
        massWeight
        """
        self.read()

        # read/compute and factorize the mass matrix for the rest shape
        if constProj_massWeight:
            # read mass file
            self.load_factorize_masses()

            # compute weighted snapshots M^{1/2} X
            assert self.snapTensor.shape[1] == self.massL.shape[0]
            self.snapTensor *= self.massL[:, None]

        if constProj_standarize:
            self.standarize()

        print("after-process stats,  min:", np.min(self.snapTensor), "max: ", np.max(self.snapTensor),
              "mean: ", np.mean(self.snapTensor), "std:", np.std(self.snapTensor))
        print('nonlinearSnapshots ready ... Volkwein ('+str(constProj_massWeight)+'), standarized ('+str(constProj_standarize)+').')

    def read(self):
        """ read separate stored constraints´ projections,
           and build frames tensor """

        print("reading the nonlinear snapshots tensor ...")
        Xtemp = []
        for i in range(self.frs):
            file = open(self.snapshots_file+str(i)+".bin", "rb")
            #  read matrix dimension
            ni = struct.unpack('<i', file.read(4))[0]   # read a 4 byte integer in little endian
            mi = struct.unpack('<i', file.read(4))[0]
            Mat_i = np.zeros((ni, mi))   # (ep, 3)  #  expected dimension of each snapshot

            for coli in range(mi):
                for rowi in range(ni):
                    value = struct.unpack('<d', file.read(8))[0]  # read 8 byte little endian double
                    Mat_i[rowi, coli] = value
            if i == 0:
                Xtemp = Mat_i[np.newaxis, :, :]    # create snapshots tensor
            #  print(Xtemp.shape)

            else:
                Xtemp = np.concatenate((Xtemp, Mat_i[np.newaxis, :, :]), axis=0)    # update snapshots tensor
            #  print(Xtemp.shape) # (F, ep, 3)

        self.constraintVerts = Xtemp.shape[1]//self.constraintsSize   # e == e.p//p
        self.snapTensor = Xtemp  # initialized with the un-pre-processed snapshots
        print("loaded snapshots size", self.snapTensor.shape)
        print("No. constrained verts: ", self.constraintVerts)
        print("pre-process stats,  min:", np.min(self.snapTensor), "max: ", np.max(self.snapTensor),
              "mean: ", np.mean(self.snapTensor), "std:", np.std(self.snapTensor) )

    def load_factorize_masses(self):
        # load  m_vertexMass for the constrained simulation
        fileMass = open(self.mass_file, "rb")   # mass matrix from the auxiliary vatriable
        ni = struct.unpack('<i', fileMass.read(4))[0]   # read a 4 byte integer in little endian
        mi = struct.unpack('<i', fileMass.read(4))[0]

        hrpdAuxiliariesMass = np.zeros((ni))
        for j in range(ni):
            value = struct.unpack('<d', fileMass.read(8))[0]  # read 8 byte as little endian double
            hrpdAuxiliariesMass[j] = value
        fileMass.close()

        #  compute Cholesky factorization for the diagonal auxliary mass matrix
        massL = np.sqrt(hrpdAuxiliariesMass)  # ep

        #  check the Cholesky factorization is done properly:
        assert(np.allclose(np.multiply(massL, massL)-hrpdAuxiliariesMass, np.zeros(ni)))  # assert: LL^T = Masses

        invMassL = np.zeros(ni)  # ep
        for j in range(ni):
            if massL[j]:
                invMassL[j] = 1/massL[j]
            else:
                invMassL[j] = 0

        #  check the inverse of the Cholesky factorization:
        assert(np.allclose(np.multiply(invMassL, massL), np.ones(ni)))  # assert: L^{-1}L = I

        self.massL = massL
        self.invMassL = invMassL
        # print("Mass matrix ready ...")

    def standarize(self):

        if self.rest_shape == "first":
            self.mean = self.snapTensor[0].copy()  # (ep, 3)  (maybe weighted) first given frame

        elif self.rest_shape == "average":
            self.mean = np.mean(self.snapTensor, axis=0)  # (ep, 3)  (maybe weighted) average

        else:
            print('Error! unknown rest shape: ', self.rest_shape)
            sys.exit(1)

        # we subtract mean and normalize w. r. to. std(snapshots) to bring data center as close as possible to zero
        # and the standard deviation as close as possible to one!

        # 1- subtract the mean value
        self.snapTensor -= self.mean[np.newaxis]  # (F, ep, 3)

        # 2- normalize snapshots
        self.pre_scale_factor = 1 / (np.std(self.snapTensor))
        self.snapTensor *= self.pre_scale_factor

