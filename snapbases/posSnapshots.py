# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases developers and contributors. All rights reserved.
# License: Apache-2.0

import struct
import sys
import os
import csv
import argparse
import h5py
import numpy as np
from numpy.linalg import matrix_rank
import scipy.linalg as spla
from scipy.linalg import svd, norm, cho_factor, cho_solve, cholesky, orth
from utils.support import GeodesicDistanceComputation
import cProfile
import pstats
import igl
root_folder = os.getcwd()
profiler = cProfile.Profile()


class posSnapshots:
    """
    Position snapshots class
    following methods defined to read a .h5 input animation file with a possiblly pre-aligned frames of size (F, N, 3),
    if required the snapshots will be further pre-processed through standerization and/or mass weighting
    """

    def __init__(self, input_animation_file, rest_shape, masses_file, standarize=True, massWeight=True):

        self.input_animation_file = input_animation_file  # contains pre-aligned (only centered) snapshots
        self.rest_shape = rest_shape  # which frame to use as rest-shape ("first" or "average")

        self.verts = None  # vertices
        self.tris = None  # faces
        self.frs = 0  # no. frames: F
        self.nVerts = 0  # no. vertices: N

        self.mean = None  # (nVerts, 3)
        self.pre_scale_factor = 1  # normalization factor
        self.massesFile = masses_file  # file contains mass weights as one vector

        self.mass = None   # vertices masses vector
        self.massL = None  # Cholesky factorization L of mass matrix Mass = L^T L
        self.invMassL = None  # Choesky factorisation inverse L^{-1}

        self.snapTensor = None  # preprocessed snapshots tensors on which we compute components (basis/ modes)

        self.compute_geodesic_distance = None  # geodesic distances computed on the average shape

        # One time compute snapTensor
        self.do_snapshots_precomputations(standarize, massWeight)

    def do_snapshots_precomputations(self, standarize, massWeight):
        """
        One time snapshots loading and possibly pre-processing. Options are:
        standarize (note: this step includes also geodesics distances computation),
        massWeight
        """

        # read (probably) aligned snapshots/frames file .h5
        self.read()
        self.snapTensor = self.verts.copy()  # initialized with the snapshots

        # read/compute and factorize the mass matrix for the rest shape
        if massWeight:
            # read mass file
            self.read_factorize_masses()

            # compute weighted snapshots M^{-1/2} X
            assert self.snapTensor.shape[1] == self.massL.shape[0]
            self.snapTensor *= self.massL[:, None]

        # prepare geodesic distance computation on the rest pos mesh for local support
        if self.rest_shape == "first":
            self.mean = self.snapTensor[0].copy()  # (N, 3)  (maybe weighted) first given frame

        elif self.rest_shape == "average":
            self.mean = np.mean(self.snapTensor, axis=0)  # (N, 3)  (maybe weighted) average

        else:
            print('Error! unknown rest shape: ', self.rest_shape)
            sys.exit(1)

        # geodesic distances are computed on non weighted shape
        if self.rest_shape == "first":
            self.compute_geodesic_distance = GeodesicDistanceComputation(self.verts[0], self.tris)
        elif self.rest_shape == "average":
            self.compute_geodesic_distance = GeodesicDistanceComputation(np.mean(self.verts, axis=0), self.tris)

        # sandarize data
        if standarize:
            self.standarize(massWeight)
        
        print('Snapshots ready... Volkwein ('+str(massWeight)+'), standarized ('+str(standarize)+').')

    def read(self):
        with h5py.File(self.input_animation_file, 'r') as f:
            self.verts = f['verts'][()].astype(float)  # (F, N, 3)
            self.tris = f['tris'][()]

        self.frs, self.nVerts, _ = self.verts.shape

        print("Vertices: ", self.nVerts)
        print("Faces: ", self.tris.shape[0])
        print("Frames: ", self.frs)

    def read_factorize_masses(self):
        # if masses file is available, read it
        fileName = self.massesFile
        N = self.nVerts

        hrpdMass = np.zeros(N)  # m_vertexMass from hrpd simulation

        if not fileName:
            # if no file given, use igl to compute masses
            m = igl.massmatrix(self.verts[0], self.tris, igl.MASSMATRIX_TYPE_VORONOI)
            hrpdMass = np.diag(m.todense())
            hrpdMass = hrpdMass / hrpdMass.sum() * 2
        else:
            try:
                with open(fileName, "rb") as fileMass:  # mass matrix (im: boxed in -0.5-0.5)
                    for j in range(N):
                        value = struct.unpack('<d', fileMass.read(8))[0]
                            # read 8 byte and interpret them as little endian double
                        hrpdMass[j] = value

                fileMass.close()
            except IOError:
                print(fileName + " could not be read")

        self.mass = hrpdMass.copy()
        massL = cholesky(np.diag(hrpdMass))  # (N, N)
        invMassL = spla.inv(massL)  # (N, N)

        self.massL = np.diagonal(massL)  # (N,)
        self.invMassL = np.diagonal(invMassL)  # (N,)

    def standarize(self, massWeight):
        # we subtract Xmean and normalize w. r. to. std(X) to bring data center as close as possible to zero
        # and the standard deviation as close as possible to one!

        # 1- subtract the mean value
        self.snapTensor -= self.mean[np.newaxis]  # (F, N, 3)

        # 2- normalize snapshots
        self.pre_scale_factor = 1 / (np.std(self.snapTensor))
        self.snapTensor *= self.pre_scale_factor

        # np.save('X_classes', self.snapTensor)

