import os
import numpy as np
import struct
import sys
import cProfile
from utils.utils import log_time, read_sparse_matrix_from_bin, read_mesh_file, read_obj
from utils.process import view_anim_file, view_components
from utils.support import compute_tetMasses, compute_triMasses, compute_edge_incidence_matrix_on_tris
import h5py
from config.config import Config_parameters
import igl
from scipy.sparse import coo_matrix
root_folder = os.getcwd()
profiler = cProfile.Profile()

constProj_output_directory = ""
class nonlinearSnapshots:
    """
    Constraints snapshots class
    read and pre-process a .bin input pre-recorded 'F' snapshots, each of size (ep, 3),
    if required the snapshots will be mass-weighted and/or standardized.
    """

    def __init__(self,param: Config_parameters):

        self.snapshots_file = ""  # contains pre-aligned (only centered) snapshots
        self.rest_shape = ""  # which frame to use as rest-shape ("first" or "average")
        self.dim = 0
        self.mass_file = ""  # file contains mass weights as one vector
        self.frs = 0  # no. frames: F
        self.constraintsSize = 0  # 'p' in the paper. no. rows in each projection mat
        self.num_constained_elements = 0  # numConstraints/'e'  (verts, tris, tets)

        self.mean = None  # (nVerts, 3)
        self.pre_scale_factor = 1  # normalization factor

        self.mass = None   # masses vector
        self.massL = None  # Cholesky factorization L of mass matrix Mass = L^T L
        self.invMassL = None  # Choesky factorisation inverse L^{-1}

        self.snapTensor = None  # preprocessed snapshots tensors on which we compute components (basis/ modes)
                                # expected size of (F, ep, 3)

        self.test_snapTensor = None  # preprocessed test snapshots tensors on which we compute components (basis/ modes)
        # expected size of (F, ep, 3)
        # Elements of the tetrahedralized mesh
        self.verts = None
        self.tris = None
        self.tets = None
        self.edges = None
        self.tet_mesh = None
        self.ele_type = ""  # str: type of constrained elements (_verts, _edges,_faces, _tets)
        self.param = param

    def config(self):
        """
            All parameters of this function are defined and can be manipulated using relative *.json and config.py
        """
        global constProj_output_directory
        self.snapshots_file = self.param.constProj_input_snapshots_pattern  # contains pre-aligned (only centered) snapshots
        self.rest_shape = self.param.constProj_rest_shape  # which frame to use as rest-shape ("first" or "average")
        self.dim = self.param.constProj_dim
        self.mass_file = self.param.constProj_masses_file  # file contains mass weights as one vector
        self.frs = self.param.constProj_numFrames  # no. frames: F
        self.constraintsSize = self.param.constProj_p_size  # 'p' in the paper. no. rows in each projection mat
        # (=3 for TetStrain constraint, =2 for tristrain, =1 otherwise)
        self.ele_type = self.param.constProj_element_type

        self.tet_mesh = self.param.tet_mesh_file
        self.tri_mesh = self.param.tri_mesh_file
        constProj_output_directory = self.param.constProj_output_directory

    @log_time(constProj_output_directory)
    def snapshots_prepare(self):
        """
        One time snapshots loading and possibly pre-processing. Options are:
        standarize (note: this step includes also geodesics distances computation),
        massWeight
        """
        self.read()

        # read/compute and factorize the mass matrix for the rest shape
        if self.param.constProj_massWeight:
            # read mass file
            self.load_factorize_masses()

            # compute weighted snapshots M^{1/2} X
            assert self.snapTensor.shape[1] == self.massL.shape[0]
            self.snapTensor *= self.massL[:, None]

        if self.param.constProj_standarize:
            self.standarize()

        print("after-process stats,  min:", np.min(self.snapTensor), "max: ", np.max(self.snapTensor),
              "mean: ", np.mean(self.snapTensor), "std:", np.std(self.snapTensor))
        print('nonlinearSnapshots ready ... Volkwein ('+str(self.param.constProj_massWeight)+'), standarized ('+str(self.param.constProj_standarize)+').')

    @log_time(constProj_output_directory)
    def read(self):
        """ read separate stored constraints´ projections,
           and build frames tensor """

        print("reading the nonlinear snapshots tensor ...")
        Xtemp = []

        for i in range(0, self.frs* self.param.constProj_frame_increment, self.param.constProj_frame_increment):
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

            else:
                Xtemp = np.concatenate((Xtemp, Mat_i[np.newaxis, :, :]), axis=0)    # update snapshots tensor
            # (F, ep, 3)

        self.num_constained_elements = Xtemp.shape[1]//self.constraintsSize   # e == e.p//p
        self.snapTensor = Xtemp  # initialized with the un-pre-processed snapshots
        print("loaded snapshots size", self.snapTensor.shape)
        print("No. constrained verts: ", self.num_constained_elements)
        print("pre-process stats,  min:", np.min(self.snapTensor), "max: ", np.max(self.snapTensor),
              "mean: ", np.mean(self.snapTensor), "std:", np.std(self.snapTensor) )
        # --------------------------------------------------------------------------------------------------------------
        X_test = [] # load test snapshots
        for i in range(self.param.constProj_train_test_jump, self.frs* self.param.constProj_frame_increment, self.param.constProj_frame_increment):
            file = open(self.snapshots_file+str(i)+".bin", "rb")
            #  read matrix dimension
            ni = struct.unpack('<i', file.read(4))[0]   # read a 4 byte integer in little endian
            mi = struct.unpack('<i', file.read(4))[0]
            Mat_i = np.zeros((ni, mi))   # (ep, 3)  #  expected dimension of each snapshot

            for coli in range(mi):
                for rowi in range(ni):
                    value = struct.unpack('<d', file.read(8))[0]  # read 8 byte little endian double
                    Mat_i[rowi, coli] = value
            if i == self.param.constProj_train_test_jump:
                X_test = Mat_i[np.newaxis, :, :]    # create snapshots tensor

            else:
                X_test = np.concatenate((X_test, Mat_i[np.newaxis, :, :]), axis=0)    # update snapshots tensor

        self.test_snapTensor = X_test
        print("loaded test snapshots size", self.test_snapTensor.shape)
        print("non-processed test stats,  min:", np.min(self.test_snapTensor), "max: ", np.max(self.test_snapTensor),
              "mean: ", np.mean(self.test_snapTensor), "std:", np.std(self.test_snapTensor))
        # --------------------------------------------------------------------------------------------------------------

        #self.store_snapshots_animations(constProj_output_directory, "nonlinear_animation.h5")
        #view_anim_file( "bign_ST.h5")
    def load_factorize_masses(self):

        if os.path.exists(self.mass_file):
            # load  m_vertexMass for the constrained simulation
            fileMass = open(self.mass_file, "rb")  # mass matrix from the auxiliary vatriable
            ni = struct.unpack('<i', fileMass.read(4))[0]  # read a 4 byte integer in little endian
            mi = struct.unpack('<i', fileMass.read(4))[0]

            hrpdAuxiliariesMass = np.zeros((ni))
            for j in range(ni):
                value = struct.unpack('<d', fileMass.read(8))[0]  # read 8 byte as little endian double
                hrpdAuxiliariesMass[j] = value
            fileMass.close()
            self.mass = hrpdAuxiliariesMass.copy()
        else:
            try:
                # # if no file given, use igl to compute masses
                if self.constraintsSize == 1:
                    self.verts, self.tris = igl.read_triangle_mesh(self.tri_mesh)
                    if self.verts is None:
                        print("ERROR: Failed to read tet mesh data.")
                        return
                    # self.edges = compute_edge_incidence_matrix_on_tris(self.tris)
                    m = igl.massmatrix(self.verts, self.tris, igl.MASSMATRIX_TYPE_VORONOI)
                    vertexMasses = np.diag(m.todense())
                    self.mass = vertexMasses
                elif self.constraintsSize == 2:
                    self.verts, self.tris = igl.read_triangle_mesh(self.tri_mesh)
                    if self.verts is None:
                        print("ERROR: Failed to read tet mesh data.")
                        return
                    # self.edges = compute_edge_incidence_matrix_on_tris(self.tris)
                    m = igl.massmatrix(self.verts, self.tris, igl.MASSMATRIX_TYPE_VORONOI)
                    vertexMasses = np.diag(m.todense())
                    self.mass = compute_triMasses(vertexMasses, self.tris, self.num_constained_elements, self.constraintsSize)
                elif self.constraintsSize == 3:
                    self.verts, self.tets, self.tris = read_mesh_file(self.tet_mesh)
                    if self.verts is None:
                        print("ERROR: Failed to read tet mesh data.")
                        return
                    # self.edges = compute_edge_incidence_matrix_on_tris(self.tris)
                    m = igl.massmatrix(self.verts, self.tets)
                    vertexMasses = np.diag(m.todense())
                    self.mass = compute_tetMasses(vertexMasses, self.tets, self.num_constained_elements, self.constraintsSize)

            except IOError:
                print(self.mass_file + " could not be read")

        #  compute Cholesky factorization for the diagonal auxliary mass matrix

        massL = np.sqrt(self.mass)  # ep

        #  check the Cholesky factorization is done properly:
        assert(np.allclose(np.multiply(massL, massL)-self.mass, np.zeros(self.num_constained_elements * self.constraintsSize)))  # assert: LL^T = Masses

        invMassL = np.zeros(self.num_constained_elements * self.constraintsSize)  # ep
        for j in range(self.num_constained_elements * self.constraintsSize):
            if massL[j]:
                invMassL[j] = 1/massL[j]
            else:
                invMassL[j] = 0

        #  check the inverse of the Cholesky factorization:
        assert(np.allclose(np.multiply(invMassL, massL), np.ones(self.num_constained_elements * self.constraintsSize)))  # assert: L^{-1}L = I

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

    def store_snapshots_animations(self, output_bases_dir, file_name):

        output_file = os.path.join(output_bases_dir, file_name)
        # output_animation = os.path.join(output_bases_dir, self.output_animation_file)

        verts, tris = igl.read_triangle_mesh(self.param.tri_mesh_file)
        St = read_sparse_matrix_from_bin(self.param.constProj_weightedSt)
        anim = []
        for l in range(self.snapTensor.shape[0]):
            anim.append(St @ self.snapTensor[l, :, :])
        anim = np.array(anim)

        # save components as animation
        with h5py.File(output_file, 'w') as f:
            f['default'] = verts
            f['tris'] = tris
            for i, c in enumerate(verts):
                f['comp%03d' % i] = c
        f.close()