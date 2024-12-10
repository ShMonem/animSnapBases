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

with open("config/config.json") as fp:
    config = json.load(fp)
fp.close()

# set data sources and parameters
name = config["object"]["mesh"]
experiment = config["object"]["experiment"]  # name of you simulations

compute_pos_bases =config["vertexPos_bases"]['computeState']["compute"]
show_profile = False
# weighted selection matrix that maps constraints projections to position space
tri_mesh_file = "input_data/" \
            + name + "/" \
            + name + ".obj"
if compute_pos_bases:
    # testing state (only decoration for file name)
    # _Released / _Debugging / _Testing
    vertPos_testing = config["vertexPos_bases"]['computeState']['testingComputations']

    if vertPos_testing == "_Debugging":
        show_profile = True

    # "first"/"avarage": used for standerization step
    vertPos_rest_shape = config["vertexPos_bases"]["rest_shape"]

    # pre alignment done to frames, can be '_centered' or '_alignedRigid'
    preAlignement = config["vertexPos_bases"]["snapshots"]["preAlignement"]

    if preAlignement == "_alignedRigid": rigid = True
    elif preAlignement == "_centered": rigid = False
    else: print("Error! unknown alignment method.")

    # number of snapshots used in computations (NO. files you have)
    vertPos_maxFrames = config["vertexPos_bases"]["snapshots"]["max_numFrames"]
    snapshots_format = config["vertexPos_bases"]["snapshots"]["format"]   # either ".off" or ".ply"
    # where snapshots are stored
    snapshots_folder = config["vertexPos_bases"]["snapshots"]["snaps_folder"]
    # where animations will be stored and found
    animation_folder = config["vertexPos_bases"]["snapshots"]["anims_folder"]

    # read .off or .ply and convert it to .h5
    snapshots_anim_ready = config["vertexPos_bases"]["snapshots"]["anim_folder_ready"]
    # visualize aligned snapshots
    visualize_snapshots = config["vertexPos_bases"]["snapshots"]["visualize_aligned_animations"]

    # number of snapshots used in computations
    vertPos_numFrames = config["vertexPos_bases"]["snapshots"]["numFrames"]
    # number of bases to be computed
    vertPos_numComponents = config["vertexPos_bases"]["pca"]["numComponents"]

    if config["vertexPos_bases"]["snapshots"]["read_all_from_first"]:
        frame_increment = 1
    else:
        frame_increment = vertPos_maxFrames//vertPos_numFrames
        assert frame_increment <= 10    # max number of frame increment

    # notice that data should be put in place so that all .py can have access too!
    input_snapshots_pattern = "input_data/" \
                              + name + "/" \
                              + experiment \
                              + "/position_snapshots/" \
                              + snapshots_folder \
                              + "/pos_*" + snapshots_format

    input_animation_dir = "input_data/" \
                          + name + "/" \
                          + experiment \
                          + "/" + animation_folder + "/"

    snapshots_animation_file = "snapshots_" \
                               + str(vertPos_numFrames) \
                               + "outOf" \
                               + str(vertPos_maxFrames)\
                               +"_Frames_" \
                               + str(frame_increment) \
                               + "_increment_" \
                               + preAlignement + ".h5"

    # note that the input .h5 for bases/components computation is the output from the snapshots algnment

    """
    1st: vertex position reduction parameters
    """
    assert config["vertexPos_bases"]["pca"]["compute"] == True

    if config["vertexPos_bases"]["splocs"]["compute"]:
        vertPos_bases_type = "SPLOCS"
    else:
        vertPos_bases_type = "PCA"

    # store singVals to file during computations: True/False
    store_vertPos_PCA_sing_val = config["vertexPos_bases"]["pca"]["store_sing_val"]

    # local support and splocs parameters (better not change them!)
    # minimum geodesic distance for support map, d_min_in splocs paper
    vertPos_smooth_min_dist = config["vertexPos_bases"]["support"]["min_dist"]
    # maximum geodesic distance for support map, d_max in splocs paper: higher --> less sparsity & faster convergence
    vertPos_smooth_max_dist = config["vertexPos_bases"]["support"]["max_dist"]

    # masses file if required to pre-process snapshots
    # if not found, then libigl is necessary to compute masses
    vertPos_masses_file = "input_data/" + name + "/" + name + "_vertPos_massMatrix.bin"

    # set bases parameters
    q_standarize, q_massWeight, q_orthogonal, q_supported = False, False, False, False
    if config['vertexPos_bases']['standarized'] == '_Standarized':  # '_Standarized'/ '_nonStandarized'
        q_standarize = True
    if config['vertexPos_bases']['massWeighted'] == '_Volkwein':     # 'Volkwein' / '_nonWeighted'
        q_massWeight = True
    if config['vertexPos_bases']['orthogonalized'] == '_Orthogonalized':  # '_Orthogonalized'/'_nonOrthogonalized'
        q_orthogonal = True
    if config['vertexPos_bases']["pca"]['supported'] == '_Local':    # '_Local'/'_Global'
        q_support = 'local'
        q_supported = True
    else:
        q_support = 'global'
        q_supported = False

    if config["vertexPos_bases"]["pca"]["store_sing_val"]:
        q_store_sing_val = True
    else:
        q_store_sing_val = False

    """
        Form the name of the storing files automatically depending on the given bases type and its characteristics
    """
    vertPos_bases_name_extention = vertPos_bases_type \
                                   + preAlignement \
                                   + config['vertexPos_bases']['massWeighted'] \
                                   + config['vertexPos_bases']['standarized'] \
                                   + config['vertexPos_bases']["pca"]['supported'] \
                                   + config['vertexPos_bases']['orthogonalized'] \
                                   + vertPos_testing

    """
    # singularvalues/animations/bases depend on 
    # - experiment
    # - pre-processing of the snapshots 
    # - number of frames used out of the max frames available
    # - jumps between the selectedframes
    # - number of bases computed
    """

    vertPos_output_directory = "results/" + name + "/" + experiment + "/q_bases/" + vertPos_bases_name_extention + \
                             "/" + str(vertPos_numFrames)+ "outOf" + str(vertPos_maxFrames)+"_Frames_/" + \
                             str(frame_increment) + "_increment_" + str(vertPos_numComponents) + preAlignement+"_bases/"

    if not os.path.exists(vertPos_output_directory):
        # Create a new directory because it does not exist
        os.makedirs(vertPos_output_directory)
        print("A directory is created to store vertex position bases!")
    else:
        print("Warning! an old the store directory already exists: \n", vertPos_output_directory,\
              "\n make sure you are not over-writing! ")

    aligned_snapshots_directory = "results/" \
                                  + name \
                                  + "/" + experiment \
                                  + "/q_snapshots_h5/"

    aligned_snapshots_animation_file = "aligned_snapshots" \
                                       + str(vertPos_numFrames) \
                                       + "outOf" + str(vertPos_maxFrames)\
                                       + "_Frames_" \
                                       + str(frame_increment) \
                                       + "_increment_" \
                                       + preAlignement \
                                       + ".h5"

    vertPos_output_animation_file = "bases_animations" \
                                    + str(vertPos_numFrames) \
                                    + "outOf" \
                                    + str(vertPos_maxFrames)\
                                    + "_Frames_" \
                                    + 'computed_' \
                                    + str(vertPos_numComponents) \
                                    + "_bases.h5"


    vertPos_output_bases_ext = "results/" \
                               + name \
                               + "/" + experiment \
                               + "/q_bases/" \
                               + vertPos_bases_name_extention \
                               + "/" + experiment \
                               + "/using_" \
                               + str(vertPos_numFrames)\
                               + "outOf"\
                               + str(vertPos_maxFrames)\
                               + "_Frames_/"

    visualize_bases = config["vertexPos_bases"]["visualize"]   # boolean
    store_bases = config["vertexPos_bases"]["store"]   # boolean

    # SPLOCS paramers
    splocs_max_itrs = config["vertexPos_bases"]["splocs"]["max_itrs"]
    splocs_admm_num_itrs = config["vertexPos_bases"]["splocs"]["admm_num_itrs"]
    splocs_lambda = config["vertexPos_bases"]["splocs"]["lambda"]
    splocs_rho = config["vertexPos_bases"]["splocs"]["rho"]


"""
    2nd: constraints projections space configurations, for nonlinear bases computation.
    Options for bases type and different properties can be modified in 
    constraintsProjection_bases_config.json 
"""

compute_constProj_bases =config["constraintProj_bases"]['computeState']["compute"]

if compute_constProj_bases:
    constProj_name = config["constraintProj_bases"]["constraintType"]["name"]
    constProj_dim = config['constraintProj_bases']['dim']
    # testing state
    constProj_testing = config["constraintProj_bases"]['computeState']['testingComputations']   # _Released / _Debugging

    if constProj_testing == "_Debugging":
        show_profile = True

    # "first"/"average": used for standerization step
    constProj_rest_shape = config["constraintProj_bases"]["rest_shape"]

    # pre alignment done to frames, can be '_centered' or '_alignedRigid'
    constProj_preAlignement = config["constraintProj_bases"]["snapshots"]["preAlignement"]

    if constProj_preAlignement == "_noAlignement": centered = True
    elif constProj_preAlignement == "_centered": centered = False
    else: print("Error! unknown alignment method for .")

    # where snapshots are stored
    constProj_snapshots_type = config["constraintProj_bases"]["constraintType"]["name"]
    constProj_snapshots_folder = config["constraintProj_bases"]["constraintType"]["snaps_folder"]
    snaps_pattern_full_p = config["constraintProj_bases"]["constraintType"]["snaps_pattern_full_p"]
    # where snapshots can be stored after pre-processing # TODO
    constProj_preprocessed_snapshots_folder = config["constraintProj_bases"]["snapshots"]["processed_snapshots_file"]
    constProj_snapshots_ready = config["constraintProj_bases"]["snapshots"]["processed_snapshots_ready"]
    # number of snapshots used in computations (NO. files you have)
    constProj_maxFrames = config["constraintProj_bases"]["snapshots"]["max_numFrames"]
    # number of snapshots used in computations
    constProj_numFrames = config["constraintProj_bases"]["snapshots"]["numFrames"]
    # number of bases to be computed
    constProj_numComponents_verts = config["constraintProj_bases"]["numComponents_verts"]
    # p: the row size of the nonlinear constraint projection is p x 3
    constProj_p_size = config["constraintProj_bases"]["constraintType"]["rowSize"]

    if config["constraintProj_bases"]["snapshots"]["read_all_from_first"]:
        constProj_frame_increment = 1
    else:
        constProj_frame_increment = constProj_numFrames//constProj_maxFrames
        assert constProj_frame_increment <= 10    # max number of frame increment

    # notice that data should be put in place so that all .py can have access too!
    constProj_input_snapshots_pattern = "input_data/" \
                                        + name + "/" \
                                        + experiment \
                                        + constProj_snapshots_folder \
                                        + constProj_snapshots_type \
                                        + snaps_pattern_full_p

    constProj_input_preprocessed_snapshots_dir = "input_data/" \
                                                 + name + "/" \
                                                 + experiment + "/" \
                                                 + constProj_preprocessed_snapshots_folder + "/"

    constProj_store_sing_val = config["constraintProj_bases"]["store_sing_val"]
    constProj_element = config["constraintProj_bases"]["constraintType"]["name"]
    constProj_bases_type = config["constraintProj_bases"]["type"]

    constProj_preprocessed_snapshots_file = "snapshots_" \
                                            + str(constProj_numFrames)\
                                            + "outOf" + str(constProj_maxFrames)\
                                            + "_Frames_" \
                                            + str(constProj_frame_increment)\
                                            + "_increment_" + constProj_preAlignement \
                                            + ".bin"

    constProj_masses_file = "input_data/" \
                            + name + "/" \
                            + name + "_" \
                            + constProj_element \
                            + "_massMatrix.bin"

    # weighted selection matrix that maps constraints projections to position space
    constProj_weightedSt = "input_data/" \
                                        + name + "/" \
                                        + experiment \
                                        + constProj_snapshots_folder \
                                        + constProj_snapshots_type +"/" \
                                        + name + "_lambdaSt_"+constProj_snapshots_type+"_weighted.bin"

    constProj_constrained_Stp0 = "input_data/" \
                           + name + "/" \
                           + experiment \
                           + constProj_snapshots_folder \
                           + constProj_snapshots_type + "/" \
                           +"St_aux_0.off"

    """
    Set necessary boolean parameters
    """
    constProj_standarize, constProj_massWeight, constProj_orthogonal, constProj_support = False, False, False, False
    if config['constraintProj_bases']['standarized'] == '_Standarized':  # '_Standarized'/ '_nonStandarized'
        constProj_standarize = True
    if config['constraintProj_bases']['massWeighted'] == '_Volkwein':     # 'Volkwein' / '_nonWeighted'
        constProj_massWeight = True
    if config['constraintProj_bases']['orthogonalized'] == '_Orthogonalized':  # '_Orthogonalized'/'_nonOrthogonalized'
        constProj_orthogonal = True
    if config['constraintProj_bases']['supported'] == '_Localized':    # '_Localized'/'_Global'
        constProj_support = 'local'
    else:
        constProj_support = 'global'

    """
        Form the name of the storing files automatically depending on the given bases type and its characteristics
    """

    constProj_bases_name_extention = constProj_bases_type \
                                     + constProj_preAlignement \
                                     + config['constraintProj_bases']['massWeighted'] \
                                     + config['constraintProj_bases']['standarized'] \
                                     + config['constraintProj_bases']['supported']\
                                     + config['constraintProj_bases']['orthogonalized'] \
                                     + constProj_testing

    constProj_output_directory = "results/" \
                                 + name + "/" \
                                 + experiment\
                                 + "/p_bases/"\
                                 + constProj_name + "/" \
                                 + constProj_bases_name_extention +  "/" \
                                 + str(constProj_numFrames) \
                                 + "outOf" \
                                 + str(constProj_numFrames) \
                                 + "_Frames/" \
                                 + str(constProj_frame_increment) \
                                 + "_increment_" + "/"

    if not os.path.exists(constProj_output_directory):
        # Create a new directory because it does not exist
        os.makedirs(constProj_output_directory)
        print("A directory is created to store nonlinear bases!")
    else:
        print("Warning! an old the store directory already exists: \n", constProj_output_directory,\
              "\n make sure you are not over-writing! ")

