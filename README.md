# Animations Snapshots Reduction Subspaces

This code generates reduced subspaces for physics-based simulations, including real-time deformable character animation.

## Algorithm Features:
- Uses a snapshots method to capture the range of deformations for a character.
  - Snapshots: Collected during a *range-of-motion* test for a character mesh.
- Provides the closest approximate solution with a user-defined size.
- Tested on *projective dynamics* simulations, which use an iterative solver for position state computation.
- Faster, more stable, and visually accurate compared to *Linear-blend skinning subspaces*.

For more details, refer to the following publication(s):
- *Improved-Projective-Dynamics-Global-Using-Snapshots-based-Reduced-Bases* SIGGRAPH23 [1st place student competition award-winning paper](https://dl.acm.org/doi/10.1145/3588028.3603665).
- *On DEIM Compatibility for Constraint Projection Reduction* **soon available**

Developed by:
- [Shaimaa Monem](https://orcid.org/0009-0008-4038-3452)
- [Max Planck Institute for Dynamics of Complex Technical Systems](https://www.mpi-magdeburg.mpg.de/2316/en), Magdeburg, Germany.

Repository:
- https://github.com/ShMonem/animSnapBases
- Moved from: https://github.com/ShMonem/Snapshots-Reduction-Subspaces-for-Projective-Dynamics

License:
- Apache-2.0 see LICENSE.md.
- This code borrwes some functions from the beautiful implementations of [SPLOCS](https://github.com/tneumann/splocs), which was published under MIT license.

Copyright holders:
- [Shaimaa Monem](https://orcid.org/0009-0008-4038-3452).

## Dependencies
- Install dependencies from `venv_requirements.txt`:
````commandline
pip install -r venv_requirements.txt 
````
- Activate the virtual environment:
  - On Linux:
    ````commandline
    source re_pd/bin/activate
    ````
  - On Windows
    ````commandline
    re_pd/Scripts/Activate
    ````

## Code Structure
Thank you for your interest in our code, which you all the fun trying it out :-). !The code id structured into main directories

| Item                  | sub-directories                       | data/code                    |
|:----------------------|:--------------------------------------|:-----------------------------|
| `input_data`          | ``<mesh>\<experiment>\FOM_snapshots`` | ``*.off/*.ply``              |
| `snapbases`           |                                       | ``posSnapshots.py``          |
|                       |                                       | ``posComponents.py``         |
|                       |                                       | ``nonlinear_snapshots.py``   |
|                       |                                       | ``constraintsComponents.py`` |
| `projective_dynamics` |                                       | ``demos``                    |
|                       |                                       | ``main.py``                  |
|                       |                                       | ``Simulators.py ``           |
|                       |                                       | ``usr_interface.json``       |
| ``utils``             |                                       | ``utils.py``                 |
|                       |                                       | ``process.py``               |
|                       |                                       | ``support.py``               |
| ``test``              |                                       | ``compare_npy_files.py``     |
| ``config``            |                                       | ``config.py``                |
|                       |                                       | ``bases_config.json``        |

1. `input_data` stores your snapshots or frames in `.off` or `.ply` format. If you use a different example or file format, update the inputs in `config/bases_config.json`.

2. To run the provided example, execute `main.py` in the root directory:
	```
 	python main.py
	```
3. In `main.py`:
  - Snapshots/frames are imported, creating a `.h5` file with aligned frames.
  - `bases` container is initialized by calling the `posComponents` class.
  - `posComponents` has an `posSnapshots` attribute that access configurations from `.json`.
  - ```
	bases = posComponents()
    ```
  -  Compute the bases using:
    ```
	bases.compute_components_store_singvalues(vertPos_output_directory)
    ```
  - Store the bases/components as `.bin` or `.npy`:
    ```
	bases.store_components_to_files(vertPos_output_directory, start, vertPos_numComponents, step, '.bin')
    ```

## Reproducibility

For vertex positions reduction, a repository [redPD](https://github.com/ShMonem/redPD) is provided in order to test and reproduce results from the mentioned paper.  
  - Kindly refer to the dedicated [README.test.md](https://github.com/ShMonem/redPD/blob/main/README.test.md). 
    It explains all steps starting from snapshots collections to bases testing.

For constarints projection reduction, the directory `projective_dynamics/` 
contains [README.pd.md](https://github.com/ShMonem/animSnapBases_mirror/blob/main/projective_dynamics/demos/README.pd.md)
that explains all steps to collect nonlinear forces snapshots and basis computations, as well as running reduced simulations.

1. Clone the repo.
2. Install the virtual environment.
3. Run the code from the root directory using `python main.py` with `splocs->"compute": "No"` and then with `splocs->"compute": "Yes"`:
    ```json
    "splocs":
    {
      "compute": "No",
    }
    ```
4. ``PCA--> compute`` is always set to ``Yes``, otherwise no computations run. 
5. Bases matrices in `.bin/.npy/.h5`formats, and a`.csv` for singular values if desired, will be stored in `results/<mesh>/<experiment>/q_bases/`.
6. After bases storage, both snapshots and bases can be visualized, this option can be modified in the ``bases_config.json``. Associated animation files can be found in both `results/<mesh>/<experiment>/q_snapshts_h5/` and `results/<mesh>/<experiment>/q_bases/`.
    Note: In order to stop the snapshots animations from the animations cycle, press ``stop animation --> OK`` then close the tap, otherwise it might produce an error.

For code navigation and further development, refer to ``config/README.dev.md``.
