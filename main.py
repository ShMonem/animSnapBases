# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import os
import cProfile
import pstats
import csv
import argparse
from utils.process import convert_sequence_to_hdf5, load_off, load_ply, align, view_anim_file, view_components
from utils.utils import store_matrix, store_vector, write_tensor_to_bin_colmajor
from functools import partial
# vertex position parameters
from config.config import Config_parameters
import time
root_folder = os.getcwd()
profiler = cProfile.Profile()


def main(param: Config_parameters):

    if param.compute_pos_bases:  # if position bases will be computed
        from snapbases.posComponents import posComponents
        print("Computing bases for positions vertices")
        # in case input_animation_dir has ot been created yet:
        # read snapshots: list of meshes in .off or .ply format
        aligned_training_snapshots_h5_file = os.path.join(param.aligned_snapshots_directory, param.train_aligned_snapshots_animation_file)
        aligned_testing_snapshots_h5_file = os.path.join(param.aligned_snapshots_directory, param.test_aligned_snapshots_animation_file)

        if not os.path.exists(aligned_training_snapshots_h5_file) or not os.path.exists(aligned_testing_snapshots_h5_file) :

            print("preparing snapshots...")
            # Create a new directory if it does not exist
            if not os.path.exists(param.input_animation_dir):
                os.makedirs(param.input_animation_dir)
                print("Directory is created to store imported snapshots animations!")
            if not os.path.exists(param.aligned_snapshots_directory):
                os.makedirs(param.aligned_snapshots_directory)
                print("Directory is created to store aligned snapshots animations!")

                print("Frame increament: ", param.frame_increment)
            train_snapsots_h5_file = os.path.join(param.input_animation_dir, param.train_snapshots_animation_file)
            test_snapsots_h5_file = os.path.join(param.input_animation_dir, param.test_snapshots_animation_file)

            if param.snapshots_format == ".off":
                # training snapshots
                convert_sequence_to_hdf5(param.input_snapshots_pattern, partial(load_off, no_colors=True),
                                         train_snapsots_h5_file, param.vertPos_numFrames, param.frame_increment)
                # testing snapshots
                convert_sequence_to_hdf5(param.input_snapshots_pattern, partial(load_off, no_colors=True),
                                         test_snapsots_h5_file, param.vertPos_numFrames,
                                         param.frame_increment+param.train_test_jump)
            elif param.snapshots_format == ".ply":
                # TODO: test
                convert_sequence_to_hdf5(param.input_snapshots_pattern, load_ply,
                                         train_snapsots_h5_file, param.vertPos_numFrames, param.frame_increment)
                convert_sequence_to_hdf5(param.input_snapshots_pattern, load_ply,
                                         test_snapsots_h5_file, param.vertPos_numFrames,
                                         param.frame_increment+param.train_test_jump)
            else:
                print("Yet, only .off/.ply mesh files are supported for snapshots!")
                return

            align(train_snapsots_h5_file, aligned_training_snapshots_h5_file, param.rigid)
            align(test_snapsots_h5_file, aligned_testing_snapshots_h5_file, param.rigid)

        else:
            print("A training and testing snapshots file already exists: \n", aligned_training_snapshots_h5_file,
                  "\n .. skip import! ")

        # read and pre-process snapshots
        bases = posComponents(param)

        # compute bases/components and store PCA singularvalues
        bases.compute_components_store_singvalues()
        bases.post_process_components()

        # store bases
        bases.store_animations(param.vertPos_output_directory)

        # visualize aligned snapshots and computed bases
        if param.visualize_snapshots:
            view_anim_file(aligned_training_snapshots_h5_file)

        if param.visualize_bases:
            view_components(os.path.join(param.vertPos_output_directory, bases.output_components_file))

        # run re-construction and analysis tests
        if param.run_pca_tests:
            from generate_figures.pos_reduction_tests import tets_plots_pca

            tets_plots_pca(bases, param)

        if param.store_bases:
            start = 1
            end = bases.numComp
            step = 1
            bases.store_components_to_files(start, end, step, ".bin")


    if param.compute_constProj_bases:
        from snapbases.constraintsComponents import constraintsComponents
        print("Computing nonlinear bases for")
        ''' Compute PCA bases/components as they are required any way!'''
        nonlinearBases = constraintsComponents(param)

        # Configuring snapsots parameters and nonliner parameters can be modified in config.json and config.py
        nonlinearBases.nonlinearSnapshots.config()
        nonlinearBases.config()

        # Read and preprocess nonlinear snapshots
        nonlinearBases.nonlinearSnapshots.snapshots_prepare()

        # Compute PCA bases for nonlinear function and store singular value if desired
        nonlinearBases.compute_components_store_singvalues()

        # Post-process bases w.r.to standardization and mass weighting
        nonlinearBases.post_process_components()

        # Compute DEIM Interpolation points
        deim_interpolation_in_pos_space = True
        nonlinearBases.deim_blocksForm(deim_interpolation_in_pos_space)

        if param.store_nonlinear_bases:
            start = 1
            end = nonlinearBases.numComp
            step = 1
            nonlinearBases.store_components_to_files(start, end, step, '.bin')

        if param.run_deim_tests:
            from generate_figures.nl_reduction_tests import tets_plots_deim
            tets_plots_deim(nonlinearBases, pca_tests= True, postProcess_tests=False,
                                            deim_tests=True, visualize_deim_elements=True)

if __name__ == '__main__':
    # -----------------------------------------------------------------------------------------------------------------
    # current available examples:
    # meshes = ["sphere", "armadillo", "bunny", "elephant", "octopus"]
    # subspaces = ["posSubspace", "tetstrainSubspace", "tristrainSubspace","vertstarbendingSubspace"]

    parser = argparse.ArgumentParser(description="Set bses parameters.")
    parser.add_argument('--mesh', type=str, default="sphere", help='Give a character mesh (default: sphere)')
    parser.add_argument('--subspace', type=str, default="posSubspace",
                        help='Subspaces for which bases are computed (default: posSubspace)')

    args = parser.parse_args()
    param = Config_parameters()
    #
    jason_file = "config/examples/"+ args.mesh +"_gFall_"+ args.subspace+".json"
    param.reset(jason_file)
    main(param)
    # -----------------------------------------------------------------------------------------------------------------
    # when snapshots of the reduced simulations is available we run more on mesh accuracy tests
    # position space
    if param.compute_pos_bases and  param.reduced_snapshots_available:
        from generate_figures.onMesh_accuracyMeasures import compute_accuracy, readOriginalMesh
        # Reconstruction
        compute_accuracy( param.tet_mesh_file, param.snapshots_format,
                          1, param.vertPos_numFrames, 50, 10, 200,
                          param.input_snapshots_files_name,
                          param.input_pos_snapshots_dir + "/posPCA_","_noConstraintProjReduction/pos_",
                          "posSubspace",
                          param.vertPos_output_directory
                          )
        # Testing om unseen data
        compute_accuracy(param.tet_mesh_file, param.snapshots_format,
                         param.vertPos_numFrames +1, param.vertPos_numFrames+50, 50, 10, 200,
                         param.input_snapshots_files_name,
                         param.input_pos_snapshots_dir + "/posPCA_", "_noConstraintProjReduction/pos_",
                         "posSubspace",
                         param.vertPos_output_directory,
                         case ="_test_on_unseen_set"
                         )


    # constraint projection space
    if param.compute_constProj_bases and param.reduced_constProj_snapshots_available:
        from generate_figures.onMesh_accuracyMeasures import compute_accuracy, readOriginalMesh
        # Reconstruction
        compute_accuracy(param.tet_mesh_file, param.snapshots_format,
                             1, param.constProj_numFrames, 10, 10, 120,
                             param._pos_snaps_folder+"/pos_",
                             param._deim_pos_snaps_folder, "_"+param.constProj_name+"/pos_",
                             "posSubspace",
                             param.vertPos_output_directory
                             )
        # Testing om unseen data
        compute_accuracy(param.tet_mesh_file, param.snapshots_format,
                         param.constProj_numFrames +1, param.constProj_numFrames+50, 10, 10, 120,
                         param._pos_snaps_folder + "/pos_",
                         param._deim_pos_snaps_folder, "_" + param.constProj_name + "/pos_",
                         "posSubspace",
                         param.vertPos_output_directory,
                         case="_test_on_unseen_set"
                         )
    # -----------------------------------------------------------------------------------------------------------------
