import os
import cProfile
import pstats
from IPDGS.classes.posComponents import posComponents
from external.splocs.inout import convert_sequence_to_hdf5, load_off, load_ply
from external.splocs.align_rigid import align
from external.splocs.view_animation import view_anim_file
from external.splocs.view_splocs import view_components
from functools import partial
from IPDGS.config.config import vertPos_numFrames, snapshots_format, frame_increament, input_snapshots_pattern, \
                                input_animation_dir, snapshots_animation_file, \
                                visualize_snapshots, vertPos_output_directory, vertPos_numComponents, aligned_snapshots_directory, aligned_snapshots_animation_file, \
                                vertPos_output_animation_file, snapshots_anim_ready, rigid


root_folder = os.getcwd()
profiler = cProfile.Profile()


def main():

    # in case input_animation_dir hasn ot been created yet:
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

            print("Frame increament: ", frame_increament)
        snapsots_h5_file = os.path.join(input_animation_dir, snapshots_animation_file)
        if snapshots_format == ".off":
            convert_sequence_to_hdf5(input_snapshots_pattern, partial(load_off, no_colors=True),
                                     snapsots_h5_file, vertPos_numFrames, frame_increament)
        elif snapshots_format == ".ply":  # TODO: test
            convert_sequence_to_hdf5(input_snapshots_pattern, load_ply,
                                     snapsots_h5_file, frame_increament)
        else:
            print("Yet, only .off/.ply mesh files are supported for snapshots!")
            return

        align(snapsots_h5_file, aligned_snapshots_h5_file, rigid)

    else:
        print("A snapshots file already exists: \n", aligned_snapshots_h5_file,
              "\n .. we skip import! ")

    if visualize_snapshots == "Yes":
        view_anim_file(aligned_snapshots_h5_file)

    # read and pre-process snapshots
    bases = posComponents()

    # compute bases/components and store PCA singularvalues
    bases.compute_components_store_singvalues(vertPos_output_directory)
    bases.post_process_components()

    # store bases
    bases.store_animations(vertPos_output_directory)
    # bases.store_components_to_files(vertPos_output_directory, vertPos_numComponents, vertPos_numComponents, 10, '.npy')
    # bases.store_components_to_files(vertPos_output_directory, 10, vertPos_numComponents, 10, '.bin')

    view_components(os.path.join(vertPos_output_directory, bases.output_components_file))
if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    # 	description='Find Deformation Bases')
    # parser.add_argument('input_animation_file')
    # parser.add_argument('output_components_file')
    # parser.add_argument('-a', '--output_anim',
    # 					help='Output animation file (will also save the component weights)')
    # args = parser.parse_args()
    # main(args.input_animation_file,
    # 	args.output_components_file,
    # 	args.output_anim)

    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()
