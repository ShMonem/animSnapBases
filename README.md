# Improved-Projective-Dynamics-Global-Using-Snapshots-based-Reduced-Bases

Implementation for position subspace computation for the SIGGRAPH23 [1st place student comptition award winning paper](https://dl.acm.org/doi/10.1145/3588028.3603665).

As an external dependency and for comparision reasons, SPLOCS algorithm with this code (not as a submodule), you can fine the [original SPLOCS here](https://github.com/tneumann/splocs)
In addition, to re-produce the results in our paper you can use implementations from the [hyper-reduced-projective-dynamics paper here](https://replicability.graphics/papers/10.1145-3197517.3201387/index.html). All you need it to replace the LBS subspaces, and read a `.bin` bases file that this code produces

Note that, this codes provides more options than it has been published in the paper yet. And venv can be re-generated from `venv_requirements.txt`
You can play with all options of othogonality, support, locality and mass weighting in the associated `bases_config.json`

## For internal code review purpose (MPI, CSC group)
A `comparision_test` and `input_data` folders are provided in order to test the outcome. There folders can be fount under `/afs/mpi-magdeburg.mpg.de/data/csc/ShaimaaMonem/publications/Snapshots_reduction_subspaces_4projective_dynamics/`
1- make sure that .json is set to the same parameters used in the paper
```
"vertexPos_bases":
		{
		"max_numFrames": 200,
		"numFrames": 200,	
		"dim": 3,
		"rest_shape": "first",
		"massWeighted": "_Volkwein",
		"standarized": "_Standarized",
		"orthogonalized": "_nonOrthogonalized",
		"support":
		{
			"min_dist": 0.1,
			"max_dist": 0.25
		},
		"PCA":
		{
			"compute": "Yes",
			"numComponents": 200,
			"supported": "_Local",
			"store_sing_val": "No"
		},
		"splocs":
		{
			"compute": "No",
			"max_itrs": 20,
			"admm_num_itrs": 10,
			"lambda": 2,
			"rho": 10.0
		},
```
2- Your ".ply" or ".off" frames should be inside `/input_data/<mesh_name>/_gravitationalFall/FOM_snapshots_OFF` in the same directory as `main.py`
3- The mass matrix, when available, should be inside `/input_data/<mesh_name>/`

2- From `comparision_test` run
```
python3 compare_npy_files.py PCA_centered_Volkwein_Standarized_Local_nonOrthogonalized_200outOf200_Frames_using_F_200K200.npy ../results/bunny/q_bases/PCA_centered_Volkwein_Standarized_Local_nonOrthogonalized_Debugging/200outOf200_Frames_/1_increament_200_centered_bases/using_F_200K200.npy
```
you should see somthing like
```
File one contains a (200, 14290, 3) tensor, and file two (200, 14290, 3)
checking if identical ... True
checking if close ... True
 testing the sparsity of a
 ... not sparse.
 testing the sparsity of b
 ... not sparse.
```
3- Similarly you can test for the SPLOCS bases, remember to change "splocs"-->"compute": "Yes" ("PCA"--> "compurte" is always a yes!)
