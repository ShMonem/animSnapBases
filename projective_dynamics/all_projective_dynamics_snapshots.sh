#!/bin/bash

## current available examples:
#meshes = ["cloth", "bar";]
# examples = ["cloth_bend", "cloth_spring", "cloth_strain", "cloth_bend_spring_strain",
#             "bar_deformationgradient", "bar_tetstrain",
#             ]

# Before envoking the code first time: chmod +x all_projective_dynamics_snapshots.sh
# Then run the examples : ./all_projective_dynamics_snapshots.sh
for _mesh in "cloth";do
  for _example in "cloth_bend_spring_strain";do
    for _gravity in "active" "inactive";do
      for _posSubsopace in "none" "lbs";do
        for _constrProjSubsopace in "none";do
          python main.py --mesh $_mesh --gravity $_gravity --example $_example --positionSubspace $_posSubsopace --constraintProjectionSubspace $_constrProjSubsopace
          wait
        done
      done
    done
  done
done