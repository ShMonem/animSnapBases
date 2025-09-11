
## Code Navigation and Further Information 

- All parameters can be changed in `config/bases_config.json`. 
  - The code expects that a directory ``input_data/<name>/<experiment>`` exists in the root directory,
    which can be changed in ``bases_config.json``. In the following example the used mesh character is 
    ``<name>= bunny``, and ``<experiment> = _gravitationalFall``:
  
	```json
	"object":
		{
		"mesh": "bunny",
		"experiment": "_gravitationalFall"
		},
	```
### Vertex Positions Basis Computation
- The folder containing the *snapshots*, in this case, is ``input_data/bunny/_gravitationalFall/FOM_snapshots_OFF``. You can change the paths in `.json`.
- Currently, the algorithm accepts snapshots in ``.off`` or ``.ply`` format only.
- The mass matrix, when available, should be inside `/input_data/<mesh_name>/`.
- Moreover, you can configure options for orthogonality, range of support, locality, and mass weighting in the associated `.json`:
    - Snapshots preAlignment: `_alignedRigid`/`_centered`
    - Bases Orthogonality: `_Orthogonalized`/`_nonOrthogonalized`
    - Average shape to standardize snapshots, rest_shape: `first`/`average`
    - Computed bases standardized: `_Standarized`/`_nonStandarized`
    - Computed bases "massWeighted": `_Volkwein`/`_nonVolkwein`
    - `max_numFrames` is the number of snapshots used in the computations.
    - `read_all_from_first` allows you to, for instance, pick the first 200 provided `snapshot_*.off`. If your `FOM_snapshots_OFF` contains 200 files and you want to use 50 only, a suitable increment to jump between files will be computed in `config.py`.

	```json
	"snapshots":
			{
			"format": ".off",
			"snaps_folder": "FOM_snapshots_OFF",
			"read_all_from_first": true,
			"anims_folder": "FOM_animations_h5",
			"preAlignement": "_alignedRigid" ,
			"anim_folder_ready": true,
			"visualize_aligned_animations": true
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
  - The option of computing `PCA` is always a `Yes`. If `supported = "_Global"`, the support range in `min_dist` and `max_dist` will not be used.
  - If you choose to `store_sing_val`, a `.csv` file containing the singular values against the number of bases will be stored.

  ```json
  	"PCA":
		{
		"compute": true,
		"numComponents": 200,
		"supported": "_Local",
		"store_sing_val": false
		},
	```
 
  - In case you would like to compute SPLOCS bases too, which are further optimized for sparsity and localization, you need to provide `max_itrs` for global optimization, and `admm_num_itrs` for the *ADMM* solver. You can also store bases in the form of `.bin` or `.npy` and visualize snapshots and bases, which can be changed in `main.py`.

	   ```json
		  "splocs":
			{
			"compute": true,
			"max_itrs": 20,
			"admm_num_itrs": 10,
			"lambda": 2,
			"rho": 10.0
			},
			"store":"True",
			"visualize":true
			},
	   ```
  - `testingComputations` is the only a string that is used in the naming of the directories to that you can track your own progress. 

  ```json
	"computeState":
	{
		"testingComputations": "_Released"
	}
  ```

### Constraints Projections Basis computation

-The configuration `.json` file includes options to compute bases for nonlinear forces, enabling faster reduced simulations. To use this feature, the following parameters must be specified:

- **Type of constraint** (e.g., bending, edge spring, strain)
- **Constrained element type** (e.g., vertices, edges, triangles, tetrahedrons)
- **Path to constraint projection snapshots**

  ```json

  "constraintType":
          {
              "name": "edge_spring",
              "elements": "_edges",
              "p_snaps_folder": "/constraint_projection/FOM/<specific_experiment>",
              "pos_snaps_folder": "/position_snapshots/FOM/<specific_experiment>",
              "geom_pos_snaps_folder": "/position_snapshots/deim/",
              "assembly_file_name": "assembly_ST.npz",   # assembly matrix file
              "assembly_key": "edge_spring",
              "snaps_pattern_full_p": "/edge_spring_p.npz",   # snapshots matix file
              "constrained_elements": "",
              "rowSize": 1     # row size of per-element projection 
          },
  ```

- **Type of basis to compute** (e.g., `pod_vectorized`, `pod`)
- **Type of interpolation** (e.g., `deim` or `geom`)
  ```json
  "basis_type": "pod_vectorized",
  "interpolation_type": "deim",
  ```
