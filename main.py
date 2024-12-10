# This file is part of the animSnapBases project (https://github.com/ShMonem/animSnapBases).
# Copyright animSnapBases Shaimaa Monem. All rights reserved.
# License: Apache-2.0

import os
import cProfile
import pstats
import csv
from utils.process import convert_sequence_to_hdf5, load_off, load_ply, align, view_anim_file, view_components
from utils.utils import store_matrix, store_vector, write_tensor_to_bin_colmajor
from functools import partial
# vertex position parameters
from config.config import compute_pos_bases, compute_constProj_bases, show_profile

if compute_pos_bases:
    from snapbases.posComponents import posComponents
    from config.config import vertPos_numFrames, snapshots_format, frame_increment, input_snapshots_pattern, \
        input_animation_dir, snapshots_animation_file, \
        visualize_snapshots, vertPos_output_directory, \
        aligned_snapshots_directory, aligned_snapshots_animation_file, \
        rigid, visualize_bases


# constraints projection parameters
if compute_constProj_bases:
    from snapbases.constraintsComponents import constraintsComponents
    from config.config import constProj_output_directory

    from generate_figures.nl_reduction_tests import plot_deim_reconstruction_errors

root_folder = os.getcwd()
profiler = cProfile.Profile()


def main():
    store_nonlinear_bases = True
    run_deim_tests = False
    if compute_pos_bases:  # if position bases will be computed

        print("Computing bases for positions vertices")
        # in case input_animation_dir has ot been created yet:
        # read snapshots: list of meshes in .off or .ply format
        aligned_snapshots_h5_file = os.path.join(aligned_snapshots_directory, aligned_snapshots_animation_file)

        if not os.path.exists(aligned_snapshots_h5_file):

            print("preparing snapshots...")
            # Create a new directory if it does not exist
            if not os.path.exists(input_animation_dir):
                os.makedirs(input_animation_dir)
                print("Directory is created to store imported snapshots animations!")
            if not os.path.exists(aligned_snapshots_directory):
                os.makedirs(aligned_snapshots_directory)
                print("Directory is created to store aligned snapshots animations!")

                print("Frame increament: ", frame_increment)
            snapsots_h5_file = os.path.join(input_animation_dir, snapshots_animation_file)
            if snapshots_format == ".off":
                convert_sequence_to_hdf5(input_snapshots_pattern, partial(load_off, no_colors=True),
                                         snapsots_h5_file, vertPos_numFrames, frame_increment)
            elif snapshots_format == ".ply":
                # TODO: test
                convert_sequence_to_hdf5(input_snapshots_pattern, load_ply,
                                         snapsots_h5_file, vertPos_numFrames, frame_increment)
            else:
                print("Yet, only .off/.ply mesh files are supported for snapshots!")
                return

            align(snapsots_h5_file, aligned_snapshots_h5_file, rigid)

        else:
            print("A snapshots file already exists: \n", aligned_snapshots_h5_file,
                  "\n .. skip import! ")

        # read and pre-process snapshots
        bases = posComponents()

        # compute bases/components and store PCA singularvalues
        bases.compute_components_store_singvalues(vertPos_output_directory)
        bases.post_process_components()

        # store bases
        bases.store_animations(vertPos_output_directory)

        # see aligned snapshots
        if visualize_snapshots:
            view_anim_file(aligned_snapshots_h5_file)

        if visualize_bases:
            view_components(os.path.join(vertPos_output_directory, bases.output_components_file))

    if compute_constProj_bases:
        print("Computing nonlinear bases for")
        ''' Compute PCA bases/components as they are required any way!'''
        nonlinearBases = constraintsComponents()

        # Configuring snapsots parameters and nonliner parameters can be modified in config.json and config.py
        nonlinearBases.nonlinearSnapshots.config()
        nonlinearBases.config()

        # Read and preprocess nonlinear snapshots
        nonlinearBases.nonlinearSnapshots.snapshots_prepare()

        # Compute PCA bases for nonlinear function and store singular value if desired
        nonlinearBases.compute_components_store_singvalues(constProj_output_directory)

        # Post-process bases w.r.to standardization and mass weighting
        nonlinearBases.post_process_components()

        # Compute DEIM Interpolation points
        deim_interpolation_in_pos_space = True
        nonlinearBases.deim_blocksForm(deim_interpolation_in_pos_space)

        if run_deim_tests:
            header = ['numPoints', 'fro_error', 'max_err', 'relative_errors_x', 'relative_errors_y',
                      'relative_errors_z']

            file_name = os.path.join(constProj_output_directory, "deim_convergence_tests")
            with open(file_name + '.csv', 'w', encoding='UTF8') as dataFile:
                writer = csv.writer(dataFile)
                writer.writerow(header)

                plot_deim_reconstruction_errors(nonlinearBases, writer)

            dataFile.close()

        if store_nonlinear_bases:
            start = nonlinearBases.numComp
            end = nonlinearBases.numComp+1
            step = 5
            nonlinearBases.store_components_to_files(constProj_output_directory, start, end,
                                                     step, nonlinearBases.comps, nonlinearBases.deim_alpha, '.bin')

            # write_tensor_to_bin_colmajor(nonlinearBases.comps.swapaxes(0,1), "bases_tensor_ep_kp_3")
            # store_vector("sphere_deim_S", nonlinearBases.deim_alpha, nonlinearBases.numComp, extension='.bin')
if __name__ == '__main__':

    if show_profile:
        profiler.enable()
        main()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats()
    else:
        main()
