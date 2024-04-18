"""
Configuration for bases computation.
Options for bases type and different properties can be changed in the "config.json"

"""

import json
import os

with open("IPDGS/config/bases_config.json") as fp:
    config = json.load(fp)
fp.close()

# testing state
vertPos_testing = config['computeState']['testingComputations']   # _Released / _Debugging

# set data sources and parameters
name = config["object"]["mesh"]
experiment = config["object"]["experiment"]     # name of you simulations  

"""
1st: vertex position reduction parameters
"""
 
vertPos_bases_type = config["vertexPos_bases"]["basesType"]  # PCA
# store singVals to file during computations: True/False
store_vertPos_PCA_sing_val = config["vertexPos_bases"]["store_sing_val"]
# "first"/"avarage": used for standerization step
vertPos_rest_shape = config["vertexPos_bases"]["rest_shape"]
# pre alignment done to frames, can be '_centered' or '_alignedRigid'
preAlignement = config["vertexPos_bases"]["preAlignement"]

vertPos_maxFrames = config["vertexPos_bases"]["max_numFrames"]            # number of snapshots used in computations
vertPos_numFrames = config["vertexPos_bases"]["numFrames"]            # number of snapshots used in computations
vertPos_numComponents = config["vertexPos_bases"]["numComponents"]    # number of bases to be computed

# local support and splocs parameters (better not change them!)

# minimum geodesic distance for support map, d_min_in splocs paper
vertPos_smooth_min_dist = config["vertexPos_bases"]["min_dist"]
# maximum geodesic distance for support map, d_max in splocs paper: higher --> less sparsity & faster convergence
vertPos_smooth_max_dist = config["vertexPos_bases"]["max_dist"]

    
# masses file if required to pre-process snapshots
# if not found, then libigl is necessary to compute masses
vertPos_masses_file = "input/" + name + "/" + name + "_vertPos_massMatrix.bin"
    
input_animation = "input/" + name + "/vertPos_frames/" + name + experiment + str(vertPos_maxFrames) + \
                  preAlignement + ".h5"

# set bases parameters
standarize, massWeight, orthogonal, support = False, False, False, False
if config['vertPos_bases_type']['standarized'] == '_Standarized':  # '_Standarized'/ '_nonStandarized'
    q_standarize = True
if config['vertPos_bases_type']['massWeighted'] == '_Volkwein':     # 'Volkwein' / '_nonWeighted'
    q_massWeight = True
if config['vertPos_bases_type']['orthogonalized'] == '_Orthogonalized':  # '_Orthogonalized'/'_nonOrthogonalized'
    q_orthogonal = True
if config['vertPos_bases_type']['supported'] == '_Localized':    # '_Localized'/'_Global'
    q_support = 'local'
else:
    q_support = 'global'

if vertPos_bases_type == "Yes":
    q_store_sing_val = True
else:
    q_store_sing_val = False

# set storage directories
# find the workspace
script_dir = os.path.dirname(os.path.abspath(__file__))

vertPos_bases_name_extention = vertPos_bases_type + preAlignement + q_massWeighted + q_standarized + q_supported + \
                               q_orthogonalized + vertPos_testing

vertPos_singVals_file =  "results/" + name + "/q_bases/" + vertPos_bases_type + preAlignement + q_massWeighted + \
                         q_standarized + q_supported + "/using_" + str(vertPos_numFrames)+ "outOf" + \
                         str(vertPos_maxFrames)+"_Frames_/"

vertPos_output_animation = "results/" + name + "/q_animationFiles/" + vertPos_bases_name_extention + "/" + experiment +\
                           "/using_" + str(vertPos_numFrames) + "outOf" + str(vertPos_maxFrames)+ "_Frames_" + \
                           'computed_' + str(vertPos_numComponents) + "_bases.h5"

vertPos_output_bases_ext = "results/" + name + +"/q_bases/" + vertPos_bases_name_extention + "/" + experiment + \
                           "/using_" + str(vertPos_numFrames)+ "outOf"+ str(vertPos_maxFrames)+ "_Frames_/"
                            #+ str(vertPos_numComponents) + "_bases.h5"
        
vertPos_singVals_dir = os.path.join(script_dir, vertPos_singVals_file)
vertPos_output_animation_dir = os.path.join(script_dir, vertPos_output_animation)
vertPos_output_bases_dir = os.path.join(script_dir,vertPos_output_bases_ext)


