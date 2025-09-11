## Generating Snapshots for `animSnapBases` Testing

### Constraint Projection Bases

- The directory `projective_dynamics/demos/` contains various `.json` examples for simple, designed projective dynamics simulations.

  - An example can be run by selecting the corresponding `.json` file in `projective_dynamics/main.py`.  
    For instance, the callback function associated with `projective_dynamics/demos/cloth_automated_bend_spring_strain.json` is triggered by the following line:
      ```python
      example = "cloth_automated_bend_spring_strain"
    
      ```
      This triggers the callback that simulates a cloth sheet with different constraints (bending, spring, strain) 
      and predefined time-varying positional constraints:
      ```python
      import demos.calbacks
      callback = demos.calbacks.cloth_automated_bend_spring_strain_callback(args, record_fom_info, params)
      ```

    - To record full-order simulation snapshots for basis computation, edit the parameters 
    in the ``.json`` file. Set the reduction flags to false and enable snapshot recording:
        ```json
         "constraint_projetions_reduction": {
            ...
            "vert_bending_reduced": false,
            "edge_spring_reduced": false,
            "tri_strain_reduced": false,
            ...
            },
        "nonlinear_snapshots": {
        "max_p_snapshots_num": <desider number>,
        "recodr_snapshots_info": true
        },
        ```
- Full-order snapshots are saved in:
``projective_dynamics/output/<mesh_name>/<json_file_or_example_name>/FOM/``,This path must also be specified in 
  the config file: ``animSnapBases/config/examples/<example_name>.json`` so that ``animSnapBases/main.py`` can locate the snapshots.

- To compute basis for edge spring projections, we specify experiment dirctories:
  ```json
  "object":
          {
      "experiment_dir": "projective_dynamics/output/",
      "mesh": "cloth",
      "volumetric": false,
      "experiment": "cloth_automated_bend_spring_strain/",
      "snap_format": ".off"
    },
  "constraintType":
          {
          "name": "edge_spring",
          "elements": "_edges",
          "p_snaps_folder": "/constraint_projection/FOM/<experiment>",
    },  
  ```
  
- Once the required bases are computed, you can visualize the reduced spring projections by updating the 
``.json`` to set the basis name, number of components, and basis directory:
```json
  "constraint_projetions_reduction": {
    "name": "deim_pod_vectorized",
    "edge_spring_reduced": true,
    "edge_spring_num_components": <desired_number>,
    ...
  "directories": {
    "output": "output/",
    "positions_basis": "",
    "geom_interpolation_basis_dir": "../results/cloth/cloth_automated_bend_spring_strain/p_bases/",
    "geom_interpolation_basis_file": "components_interpol_alphas_interpol_verts_interpol_alpha_ranges.npz",
```
  
  
