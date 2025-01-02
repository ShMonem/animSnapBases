# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

"""
Configuration for bases computation.
Options for bases type and different properties can be changed in the "config.json"s

"""

import json
import os

"""
    1st: position space configurations
    Options for bases type and different properties can be modified in 
    config.json 
"""


class Config_parameters:
    def __init__(self):
        # set data sources and parameters
        self.name = ""
        self.experiment = ""  # name of you simulations

        self.compute_pos_bases = False
        self.show_profile = False
        # weighted selection matrix that maps constraints projections to position space
        self.tet_mesh_file = ""
        self.tri_mesh_file = ""

        # testing state (only decoration for file name)
        # _Released / _Debugging / _Testing
        self.vertPos_testing = ""
        self.show_profile = True

        # "first"/"avarage": used for standerization step
        self.vertPos_rest_shape = ""
        self.rigid = False
        # pre alignment done to frames, can be '_centered' or '_alignedRigid'
        self.preAlignement = ""

        # number of snapshots used in computations (NO. files you have)
        self.vertPos_maxFrames = -1
        self.snapshots_format = ""  # either ".off" or ".ply"
        # where snapshots are stored
        self.snapshots_folder = ""
        # where animations will be stored and found
        self.animation_folder = ""

        # read .off or .ply and convert it to .h5
        self.snapshots_anim_ready = False
        # visualize aligned snapshots
        self.visualize_snapshots = False

        # number of snapshots used in computations
        self.vertPos_numFrames = -1
        # number of bases to be computed
        self.vertPos_numComponents = -1
        self.frame_increment = -1

        # notice that data should be put in place so that all .py can have access too!
        self.input_snapshots_pattern = ""

        self.input_animation_dir = ""

        self.snapshots_animation_file = ""
        self.vertPos_bases_type = ""
        # note that the input .h5 for bases/components computation is the output from the snapshots algnment

        # store singVals to file during computations: True/False
        self.store_vertPos_PCA_sing_val = False

        # local support and splocs parameters (better not change them!)
        # minimum geodesic distance for support map, d_min_in splocs paper
        self.vertPos_smooth_min_dist = -1
        # maximum geodesic distance for support map, d_max in splocs paper: higher --> less sparsity & faster convergence
        self.vertPos_smooth_max_dist = -1

        # masses file if required to pre-process snapshots
        # if not found, then libigl is necessary to compute masses
        self.vertPos_masses_file = ""
        self.q_standarize = False
        self.q_massWeight = False
        self.q_orthogonal = False
        self.q_support = ''
        self.q_supported = False
        self.q_store_sing_val = False

        """
        Form the name of the storing files automatically depending on the given bases type and its characteristics
        """
        self.vertPos_bases_name_extention = ""

        self.vertPos_output_directory = ""

        self.aligned_snapshots_directory = ""

        self.aligned_snapshots_animation_file = ""

        self.vertPos_output_animation_file = ""

        self.vertPos_output_bases_ext = ""

        self.visualize_bases = False  # boolean
        self.store_bases = False  # boolean

        # SPLOCS paramers
        self.splocs_max_itrs = -1
        self.splocs_admm_num_itrs = -1
        self.splocs_lambda = -1
        self.splocs_rho = -1
        self.run_pca_tests = False

        """
        2nd: constraints projections space configurations, for nonlinear bases computation.
        Options for bases type and different properties can be modified in 
        constraintsProjection_bases_config.json 
        """

        self.compute_constProj_bases = False

        self.constProj_name = ""
        self.constProj_element_type = ""

        self.constProj_dim = -1
        # testing state
        self.constProj_testing = ""  # _Released / _Debugging
        self.show_profile = False

        # "first"/"average": used for standerization step
        self.constProj_rest_shape = ""

        # pre alignment done to frames, can be '_centered' or '_alignedRigid'
        self.constProj_preAlignement = ""
        self.centered = False

        # where snapshots are stored
        self.constProj_snapshots_type = ""
        self.constProj_snapshots_folder = ""
        self.snaps_pattern_full_p = ""
        # where snapshots can be stored after pre-processing # TODO
        self.constProj_preprocessed_snapshots_folder = ""
        self.constProj_snapshots_ready = False
        # number of snapshots used in computations (NO. files you have)
        self.constProj_maxFrames = -1
        # number of snapshots used in computations
        self.constProj_numFrames = -1
        # tolerance used to satisfy bases computation criterion
        self.bases_R_tol = -1
        # p: the row size of the nonlinear constraint projection is p x 3
        self.constProj_p_size = -1
        # in deim algorithm we can choose to include a limited number of constained elements per large deformation vertex
        self.deim_ele_per_vert = -1
        self.constProj_frame_increment = -1

        # notice that data should be put in place so that all .py can have access too!
        self.constProj_input_snapshots_pattern = ""

        self.constProj_input_preprocessed_snapshots_dir = ""

        self.constProj_store_sing_val = False
        self.constProj_element = ""
        self.constProj_bases_type = ""

        self.constProj_preprocessed_snapshots_file = ""

        self.constProj_masses_file = ""

        # weighted selection matrix that maps constraints projections to position space
        self.constProj_weightedSt = ""

        self.constProj_constrained_Stp0 = ""
        self.constProj_standarize, self.constProj_massWeight, self.constProj_orthogonal, self.constProj_support = False, False, False, 'global'

        """
            Form the name of the storing files automatically depending on the given bases type and its characteristics
        """

        self.constProj_bases_name_extention = ""

        self.constProj_output_directory = ""

        self.store_nonlinear_bases = False
        self.run_deim_tests = False
        self.visualize_deim_elements = False
        self.visualize_deim_elements_at_K = False

    def reset(self, jason_file):
        with open(jason_file) as fp:
            config = json.load(fp)
        fp.close()

        # set data sources and parameters
        self.name = config["object"]["mesh"]
        self.experiment = config["object"]["experiment"]  # name of you simulations

        self.compute_pos_bases =config["vertexPos_bases"]['computeState']["compute"]
        # weighted selection matrix that maps constraints projections to position space
        self.tet_mesh_file = "input_data/" \
                    + self.name + "/" \
                    + self.name + "_made_tet.mesh"
        self.tri_mesh_file = "input_data/" \
                    + self.name + "/" \
                    + self.name + ".obj"
        if self.compute_pos_bases:
            # testing state (only decoration for file name)
            # _Released / _Debugging / _Testing
            self.vertPos_testing = config["vertexPos_bases"]['computeState']['testingComputations']

            if self.vertPos_testing == "_Debugging":
                self.show_profile = True

            # "first"/"avarage": used for standerization step
            self.vertPos_rest_shape = config["vertexPos_bases"]["rest_shape"]

            # pre alignment done to frames, can be '_centered' or '_alignedRigid'
            self.preAlignement = config["vertexPos_bases"]["snapshots"]["preAlignement"]

            if self.preAlignement == "_alignedRigid": self.rigid = True
            elif self.preAlignement == "_centered": self.rigid = False
            else: print("Error! unknown alignment method.")

            # number of snapshots used in computations (NO. files you have)
            self.vertPos_maxFrames = config["vertexPos_bases"]["snapshots"]["max_numFrames"]
            self.snapshots_format = config["vertexPos_bases"]["snapshots"]["format"]   # either ".off" or ".ply"
            # where snapshots are stored
            self.snapshots_folder = config["vertexPos_bases"]["snapshots"]["snaps_folder"]
            # where animations will be stored and found
            self.animation_folder = config["vertexPos_bases"]["snapshots"]["anims_folder"]

            # read .off or .ply and convert it to .h5
            self.snapshots_anim_ready = config["vertexPos_bases"]["snapshots"]["anim_folder_ready"]
            # visualize aligned snapshots
            self.visualize_snapshots = config["vertexPos_bases"]["snapshots"]["visualize_aligned_animations"]

            # number of snapshots used in computations
            self.vertPos_numFrames = config["vertexPos_bases"]["snapshots"]["numFrames"]
            # number of bases to be computed
            self.vertPos_numComponents = config["vertexPos_bases"]["pca"]["numComponents"]


            if config["vertexPos_bases"]["snapshots"]["read_all_from_first"]:
                self.frame_increment = 1
            else:
                self.frame_increment = self.vertPos_maxFrames//self.vertPos_numFrames
                assert self.frame_increment <= 10    # max number of frame increment

            # notice that data should be put in place so that all .py can have access too!
            self.input_snapshots_pattern = "input_data/" \
                                      + self.name + "/" \
                                      + self.experiment \
                                      + "/position_snapshots/" \
                                      + self.snapshots_folder \
                                      + "/pos_*" + self.snapshots_format

            self.input_animation_dir = "input_data/" \
                                  + self.name + "/" \
                                  + self.experiment \
                                  + "/" + self.animation_folder + "/"

            self.snapshots_animation_file = "snapshots_" \
                                       + str(self.vertPos_numFrames) \
                                       + "outOf" \
                                       + str(self.vertPos_maxFrames)\
                                       +"_Frames_" \
                                       + str(self.frame_increment) \
                                       + "_increment_" \
                                       + self.preAlignement + ".h5"

            # note that the input .h5 for bases/components computation is the output from the snapshots algnment

            """
            1st: vertex position reduction parameters
            """
            assert config["vertexPos_bases"]["pca"]["compute"] == True

            if config["vertexPos_bases"]["splocs"]["compute"]:
                self.vertPos_bases_type = "SPLOCS"
            else:
                self.vertPos_bases_type = "PCA"

            # store singVals to file during computations: True/False
            self.store_vertPos_PCA_sing_val = config["vertexPos_bases"]["pca"]["store_sing_val"]

            # local support and splocs parameters (better not change them!)
            # minimum geodesic distance for support map, d_min_in splocs paper
            self.vertPos_smooth_min_dist = config["vertexPos_bases"]["support"]["min_dist"]
            # maximum geodesic distance for support map, d_max in splocs paper: higher --> less sparsity & faster convergence
            self.vertPos_smooth_max_dist = config["vertexPos_bases"]["support"]["max_dist"]

            # masses file if required to pre-process snapshots
            # if not found, then libigl is necessary to compute masses
            self.vertPos_masses_file = "input_data/" + self.name + "/" + self.name + "_vertPos_massMatrix.bin"

            # set bases parameters
            if config['vertexPos_bases']['standarized'] == '_Standarized':  # '_Standarized'/ '_nonStandarized'
                self.q_standarize = True
            if config['vertexPos_bases']['massWeighted'] == '_Volkwein':     # 'Volkwein' / '_nonWeighted'
                self.q_massWeight = True
            if config['vertexPos_bases']['orthogonalized'] == '_Orthogonalized':  # '_Orthogonalized'/'_nonOrthogonalized'
                self.q_orthogonal = True
            if config['vertexPos_bases']["pca"]['supported'] == '_Local':    # '_Local'/'_Global'
                self.q_support = 'local'
                self.q_supported = True
            else:
                self.q_support = 'global'
                self.q_supported = False

            if config["vertexPos_bases"]["pca"]["store_sing_val"]:
                self.q_store_sing_val = True
            else:
                self.q_store_sing_val = False

            """
                Form the name of the storing files automatically depending on the given bases type and its characteristics
            """
            self.vertPos_bases_name_extention = self.vertPos_bases_type \
                                           + self.preAlignement \
                                           + config['vertexPos_bases']['massWeighted'] \
                                           + config['vertexPos_bases']['standarized'] \
                                           + config['vertexPos_bases']["pca"]['supported'] \
                                           + config['vertexPos_bases']['orthogonalized'] \
                                           + self.vertPos_testing

            """
            # singularvalues/animations/bases depend on 
            # - experiment
            # - pre-processing of the snapshots 
            # - number of frames used out of the max frames available
            # - jumps between the selectedframes
            # - number of bases computed
            """

            self.vertPos_output_directory = "results/" + self.name + "/" + self.experiment + "/q_bases/" + self.vertPos_bases_name_extention + \
                                     "/" + str(self.vertPos_numFrames)+ "outOf" + str(self.vertPos_maxFrames)+"_Frames/" + \
                                     str(self.frame_increment) + "_increment_/"

            if not os.path.exists(self.vertPos_output_directory):
                # Create a new directory because it does not exist
                os.makedirs(self.vertPos_output_directory)
                print("A directory is created to store vertex position bases!")
            else:
                print("Warning! an old the store directory already exists: \n", self.vertPos_output_directory,\
                      "\n make sure you are not over-writing! ")

            self.aligned_snapshots_directory = "results/" \
                                          + self.name \
                                          + "/" + self.experiment \
                                          + "/q_snapshots_h5/"

            self.aligned_snapshots_animation_file = "aligned_snapshots" \
                                               + str(self.vertPos_numFrames) \
                                               + "outOf" + str(self.vertPos_maxFrames)\
                                               + "_Frames_" \
                                               + str(self.frame_increment) \
                                               + "_increment_" \
                                               + self.preAlignement \
                                               + ".h5"

            self.vertPos_output_animation_file = "bases_animations" \
                                            + str(self.vertPos_numFrames) \
                                            + "outOf" \
                                            + str(self.vertPos_maxFrames)\
                                            + "_Frames_" \
                                            + 'computed_' \
                                            + str(self.vertPos_numComponents) \
                                            + "_bases.h5"


            self.vertPos_output_bases_ext = "results/" \
                                       + self.name \
                                       + "/" + self.experiment \
                                       + "/q_bases/" \
                                       + self.vertPos_bases_name_extention \
                                       + "/" + self.experiment \
                                       + "/using_" \
                                       + str(self.vertPos_numFrames)\
                                       + "outOf"\
                                       + str(self.vertPos_maxFrames)\
                                       + "_Frames_/"

            self.visualize_bases = config["vertexPos_bases"]["visualize"]   # boolean
            self.store_bases = config["vertexPos_bases"]["store"]   # boolean

            # SPLOCS paramers
            self.splocs_max_itrs = config["vertexPos_bases"]["splocs"]["max_itrs"]
            self.splocs_admm_num_itrs = config["vertexPos_bases"]["splocs"]["admm_num_itrs"]
            self.splocs_lambda = config["vertexPos_bases"]["splocs"]["lambda"]
            self.splocs_rho = config["vertexPos_bases"]["splocs"]["rho"]

            self.run_pca_tests = config["vertexPos_bases"]["run_tests"]


        """
            2nd: constraints projections space configurations, for nonlinear bases computation.
            Options for bases type and different properties can be modified in 
            constraintsProjection_bases_config.json 
        """

        self.compute_constProj_bases =config["constraintProj_bases"]['computeState']["compute"]

        if self.compute_constProj_bases:
            self.constProj_name = config["constraintProj_bases"]["constraintType"]["name"]
            self.constProj_element_type = config["constraintProj_bases"]["constraintType"]["elements"]

            self.constProj_dim = config['constraintProj_bases']['dim']
            # testing state
            self.constProj_testing = config["constraintProj_bases"]['computeState']['testingComputations']   # _Released / _Debugging

            if self.constProj_testing == "_Debugging":
                self.show_profile = True

            # "first"/"average": used for standerization step
            self.constProj_rest_shape = config["constraintProj_bases"]["rest_shape"]

            # pre alignment done to frames, can be '_centered' or '_alignedRigid'
            self.constProj_preAlignement = config["constraintProj_bases"]["snapshots"]["preAlignement"]

            if self.constProj_preAlignement == "_noAlignement": centered = True
            elif self.constProj_preAlignement == "_centered": centered = False
            else: print("Error! unknown alignment method for .")

            # where snapshots are stored
            self.constProj_snapshots_type = config["constraintProj_bases"]["constraintType"]["name"]
            self.constProj_snapshots_folder = config["constraintProj_bases"]["constraintType"]["snaps_folder"]
            self.snaps_pattern_full_p = config["constraintProj_bases"]["constraintType"]["snaps_pattern_full_p"]
            # where snapshots can be stored after pre-processing # TODO
            self.constProj_preprocessed_snapshots_folder = config["constraintProj_bases"]["snapshots"]["processed_snapshots_file"]
            self.constProj_snapshots_ready = config["constraintProj_bases"]["snapshots"]["processed_snapshots_ready"]
            # number of snapshots used in computations (NO. files you have)
            self.constProj_maxFrames = config["constraintProj_bases"]["snapshots"]["max_numFrames"]
            # number of snapshots used in computations
            self.constProj_numFrames = config["constraintProj_bases"]["snapshots"]["numFrames"]
            # tolerance used to satisfy bases computation criterion
            self.bases_R_tol = config["constraintProj_bases"]["bases_res_tol"]
            # p: the row size of the nonlinear constraint projection is p x 3
            self.constProj_p_size = config["constraintProj_bases"]["constraintType"]["rowSize"]
            # in deim algorithm we can choose to include a limited number of constained elements per large deformation vertex
            self.deim_ele_per_vert = config["constraintProj_bases"]["max_element_per_deim_vert"]

            if config["constraintProj_bases"]["snapshots"]["read_all_from_first"]:
                self.constProj_frame_increment = 1
            else:
                self.constProj_frame_increment = self.constProj_numFrames//self.constProj_maxFrames
                assert self.constProj_frame_increment <= 10    # max number of frame increment

            # notice that data should be put in place so that all .py can have access too!
            self.constProj_input_snapshots_pattern = "input_data/" \
                                                + self.name + "/" \
                                                + self.experiment \
                                                + self.constProj_snapshots_folder \
                                                + self.constProj_snapshots_type \
                                                + self.snaps_pattern_full_p

            self.constProj_input_preprocessed_snapshots_dir = "input_data/" \
                                                         + self.name + "/" \
                                                         + self.experiment + "/" \
                                                         + self.constProj_preprocessed_snapshots_folder + "/"

            self.constProj_store_sing_val = config["constraintProj_bases"]["store_sing_val"]
            self.constProj_element = config["constraintProj_bases"]["constraintType"]["name"]
            self.constProj_bases_type = config["constraintProj_bases"]["type"]

            self.constProj_preprocessed_snapshots_file = "snapshots_" \
                                                    + str(self.constProj_numFrames)\
                                                    + "outOf" + str(self.constProj_maxFrames)\
                                                    + "_Frames_" \
                                                    + str(self.constProj_frame_increment)\
                                                    + "_increment_" + self.constProj_preAlignement \
                                                    + ".bin"

            self.constProj_masses_file = "input_data/" \
                                    + self.name + "/" \
                                    + self.name + "_" \
                                    + self.constProj_element \
                                    + "_massMatrix.bin"

            # weighted selection matrix that maps constraints projections to position space
            self.constProj_weightedSt = "input_data/" \
                                                + self.name + "/" \
                                                + self.experiment \
                                                + self.constProj_snapshots_folder \
                                                + self.constProj_snapshots_type +"/" \
                                                + self.name + "_lambdaSt_weighted.bin"

            self.constProj_constrained_Stp0 = "input_data/" \
                                   + self.name + "/" \
                                   + self.experiment \
                                   + self.constProj_snapshots_folder \
                                   + self.constProj_snapshots_type + "/" \
                                   +"St_aux_0.off"

            """
            Set necessary boolean parameters
            """
            if config['constraintProj_bases']['standarized'] == '_Standarized':  # '_Standarized'/ '_nonStandarized'
                self.constProj_standarize = True
            if config['constraintProj_bases']['massWeighted'] == '_Volkwein':     # 'Volkwein' / '_nonWeighted'
                self.constProj_massWeight = True
            if config['constraintProj_bases']['orthogonalized'] == '_Orthogonalized':  # '_Orthogonalized'/'_nonOrthogonalized'
                self.constProj_orthogonal = True
            if config['constraintProj_bases']['supported'] == '_Localized':    # '_Localized'/'_Global'
                self.constProj_support = 'local'
            else:
                self.constProj_support = 'global'

            """
                Form the name of the storing files automatically depending on the given bases type and its characteristics
            """

            self.constProj_bases_name_extention = self.constProj_bases_type \
                                             + self.constProj_preAlignement \
                                             + config['constraintProj_bases']['massWeighted'] \
                                             + config['constraintProj_bases']['standarized'] \
                                             + config['constraintProj_bases']['supported']\
                                             + config['constraintProj_bases']['orthogonalized'] \
                                             + self.constProj_testing

            self.constProj_output_directory = "results/" \
                                         + self.name + "/" \
                                         + self.experiment\
                                         + "/p_bases/"\
                                         + self.constProj_name + "/" \
                                         + self.constProj_bases_name_extention +  "/" \
                                         + str(self.constProj_numFrames) \
                                         + "outOf" \
                                         + str(self.constProj_numFrames) \
                                         + "_Frames/" \
                                         + str(self.constProj_frame_increment) \
                                         + "_increment_" + "/"

            if not os.path.exists(self.constProj_output_directory):
                # Create a new directory because it does not exist
                os.makedirs(self.constProj_output_directory)
                print("A directory is created to store nonlinear bases!")
            else:
                print("Warning! an old the store directory already exists: \n", self.constProj_output_directory,\
                      "\n make sure you are not over-writing! ")

            self.store_nonlinear_bases = config["constraintProj_bases"]["store_to_files"]
            self.run_deim_tests = config["constraintProj_bases"]["run_tests"]
            self.visualize_deim_elements = config["constraintProj_bases"]["visualize_deim_elements"]
            self.visualize_deim_elements_at_K = config["constraintProj_bases"]["visualize_elements_at_bases_num"]

        print("Parameters have been resetusing:", jason_file, "!")