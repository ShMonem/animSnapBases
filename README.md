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
- Apache-2.0 see LICENSE.md.


## Code Structure for Use and Further Development
Thank you for your intrest in our code, whish you all the fun trying it out :-). !The code id structured into main directories

| Item              | Contains                                   | python files         |
| :---------------- | :---------------------------------------- | :------------------- |
| input_data        | bunny\_gravitationalFall\FOM_snapshots_OFF | *.off                |
| IPDGS             | 1- Classes:                                | posSnapshots.py      |
|                   |                                            | posComponents.py     |
|                   | 2- Config:                                 | config.py            |
|                   |                                            | bases_config.json    |
| utils             |                                            | utils.py             |
|                   |                                            | process.py           |
|                   |                                            | support.py           |
| test              |                                            | compare_npy_files.py |

1- `input_data` is where you store your snapshots or frames, in either `.off` or `.ply` format. When you run the code for a different example `<mesh>/<experiment>` than the provided `bunny\_gravitationalFall` or you change the file format, make sure you change the according inputs in `IPDGS/config/bases_config.json` where all parameters can be handeled. See *Getting started*. Your changes then are reflected directly in `IPDGS/config/config.py` where configrations is fed to different parameters called directly in the code, and the full names of the directories where results will be stored are created.

2- To run the provided example, all you need to run is the `main.py` in the root directory though
	```
 	python3 main.py
	```
3- As you can see in `main.py`:
  - First, your shapshots/frames are imported and one `.h5` file format containing the aligned frames is created
  - Then, `bases` container is initialized by calling the class `posComponents`
  - `posComponents` has an `posSnapshots` , which accesses all the choices you fed to the `.json` regarding snapshots `pre-alignment`, `weighting`, ... (now you snapshots are ready)
    ```
	bases = posComponents()
    ```
  - Then, the bases are computed through the follwing line, which again gets information the number of bases/components required and their desired properties as `locality`, `orthogonality` and so on from `.json`
    ```
	bases.compute_components_store_singvalues(vertPos_output_directory)
    ```
  - Finally you can choose wether you like to store the bases/components as matrices in the `.bin` or `.npy` format. 
    ```
	bases.store_components_to_files(vertPos_output_directory, start, vertPos_numComponents, step, '.bin')
    ```
    
## Getting Started
- Dependencies as `numpy`, `scipy` and `libigl` can be installed directly to a virtual env from `venv_requirements.txt`.
- All parameters can be changed in `IPDGS\config\bases_config.json`. 
  - The code expects that a directory `input_data/<name>/<experiment>` exsits in the root folder, which can be changed in `bases_config.json`. In the following example the used mesh character is`<name>= bunny`, and `<experiment> = _gravitationalFall`:

	```json
	"object":
		{
		"mesh": "bunny",
		"experiment": "_gravitationalFall"
		},
	```

  - The folder containg the *snapshots*, in this case is `input_data/bunny/_gravitationalFall/FOM_snapshots_OFF`, you can change the pathes in `.json`
  - Currently, the algorithim accepts snapshots in `.off` or `.ply` format only
  - The mass matrix, when available, should be inside `/input_data/<mesh_name>/`
  - Moreover, You can play with all options of othogonality, range of support, locality and mass weighting in the associated `.json`
      - Snapshots preAlignement: `_alignedRigid`/`_centered`
      - Bases Orthogonality: `_Orthogonalized`/`_nonOrthogonalized`
      - Ararge shape to standerize snapshots, rest_shape: `first`/àvarage`
      - Computed bases standarized: `_Standarized`/`_nonStandarized`
      - Computed bases "massWeighted": `_Volkwein`/`_nonVolkwein`
      - `max_numFrames` is the number od snapshots used in the computations
      - `read_all_from_first` gives you the flexibility to, for instance, pick the first 200 provided `snapshot_*.off`, otherwise if your `FOM_snapshots_OFF` contains 200 files and you want to use 50 only, then in `config.py` a suitable increament to jump between files will be computed.

	```json
	"snapshots":
			{
			"format": ".off",
			"snaps_folder": "FOM_snapshots_OFF",
			"read_all_from_first": "Yes",
			"anims_folder": "FOM_animations_h5",
			"preAlignement": "_alignedRigid" ,
			"anim_folder_ready": "Yes",
			"visualize_aligned_animations": "True"
			},
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
	```
    - The option of computing `PCA`is always a `Yes`. If `supported = "_Global"` support range in `min_dist` and `max_dist` will not be used.
    - If you choose to `store_sing_val`, a `.cvs` file containing the singular values againest the number of bases will be stored.
	```json
  	"PCA":
		{
		"compute": "Yes",
		"numComponents": 200,
		"supported": "_Local",
		"store_sing_val": "No"
		},
	```
 
  - In case you would like to compute splocs bases too, which is further optimized for sparsity and localization, you need to provide `max_itrs` for global optimization, and `admm_num_itrs` for the *ADMM* solver. Of course you can also store bases in form of `.bin` or `.npy` also visualize snapshots and bases, which can be changed in `main.py`
  
	   ```json
		  "splocs":
			{
			"compute": "Yes",
			"max_itrs": 20,
			"admm_num_itrs": 10,
			"lambda": 2,
			"rho": 10.0
			},
			"store":"True",
			"visualize":"True"
			},
	   ```
  - The `testingComputations` is only a string that is used in the naming of the directories to that you can track your own progress. 

  ```json
	"computeState":
	{
		"testingComputations": "_Released"
	}
  ```
  
## Testing the code
A `comparision_test` and `input_data` folders are provided in order to test the outcome for very few number of snapshots only for the user convenience. 
1- clone the repo
2- Install the virtual env
3- Run the code from the root directory using `python main.py`, one time with `splocs->"compute": "No"`, and one more time with `splocs->"compute": "Yes"`
```json
  "splocs":
	{
	"compute": "No",
```
4 - From the root directory run a test for the computed PCA bases
```
python3 test\compare_npy_files.py test\PCA_using_F_50K200.npy results\bunny\q_bases\PCA_alignedRigid_Volkwein_Standarized_Local_nonOrthogonalized_Release\50outOf50_Frames_\1_increament_200_alignedRigid_bases\using_F_50K200.npy
```
and/or run a test for the SPLOCS bases
```
python3 test\compare_npy_files.py test\SPLOCS_using_F_50K200.npy results\bunny\q_bases\SPLOCS_alignedRigid_Volkwein_Standarized_Local_nonOrthogonalized_Release\50outOf50_Frames_\1_increament_200_alignedRigid_bases\using_F_50K200.npy 
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
5- Note: `"PCA"--> "compurte"` is always a yes!

