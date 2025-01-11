#!/bin/bash

## current available examples:
#meshes = ["sphere", "armadillo", "elephant_normalized", "octopus"]
#subspaces = ["posSubspace", "tetstrainSubspace", "tristrainSubspace","vertstarbendingSubspace"]

for _mesh in "sphere" "armadillo" "elephant_normalized" "octopus";do
  for _subspace in "posSubspace" "tetstrainSubspace" "tristrainSubspace" "vertstarbendingSubspace";do
    python main.py --mesh $_mesh --subspace $_subspace
    wait
  done
done