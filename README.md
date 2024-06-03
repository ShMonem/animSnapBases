# Snapshots Reduction Subspaces for Projective Dynamics

The provided code can be used to generate reduced subspaces in physics based simulations for (and not limited to) deformable character animation in real-time applications. 

Algorithm Features:
- Is a snapshots method, and containes information on the possible range of deformation allowed for the character.
   - Snapshots: Possible deformations required for a character mesh. Can be collected during the necessary *range-of-motion* test done for a character before applications.
- Hence, provides the closest most accurate approximate solution with the desired *user-defined* size.
- Tested on *projective dynamics* simulations, which requires an iterative solver for positions state computation.
- Faser, numerically stable and much improved visual accuracy compared to *Linear-blend skinning subspaces*.

For more details on theory, applications and results for this work we refere to the following publication(s):
- Implementation for position subspace computation for *Improved-Projective-Dynamics-Global-Using-Snapshots-based-Reduced-Bases* the SIGGRAPH23 [1st place student comptition award winning paper](https://dl.acm.org/doi/10.1145/3588028.3603665).

Developed by:
- [Shaimaa Monem](https://orcid.org/0009-0008-4038-3452)
- [Max Planck Institute for Dynamics of Complex Technical Systems](https://www.mpi-magdeburg.mpg.de/2316/en), Magdeburg, Germany.

Repository:
- https://github.com/ShMonem/Snapshots-Reduction-Subspaces-for-Projective-Dynamics

License:
- see LICENSE.md.

## Getting Started
1- Dependencies as `numpy`, `scipy` and `libigl` can be installed directly to a virtual env from `venv_requirements.txt`.
2- All parameters can be changed in `IPDGS\config\bases_config.json`. You can play with all options of othogonality, support, locality and mass weighting in the associated ``

```json
Here goes your json object definition
```



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
