
import os
import cProfile
import pstats
from IPDGS.classes.posComponents import posComponents

root_folder = os.getcwd()
profiler = cProfile.Profile()



def main():


	bases = posComponents(vertPos_bases_type, K, vertPos_smooth_min_dist, vertPos_smooth_max_dist, output_components_file, output_animation_file,
		                  input_animation_file, vertPos_rest_shape, None, standarized, standarize, massWeighted,
		                  massWeight, orthogonalized, orthogonal, supported, support,
		                  testingComputations, store_sing_val)


   
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
