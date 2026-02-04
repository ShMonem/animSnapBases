#!/bin/bash

## current available runs:
## Uncomment all

#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_bend" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"bend":"../results/cloth/verts_bending_wi0.001_poking_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release187_Frames_1_increment/"}'\
#        --bending_Subspace_dim 1
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_bend" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"bend":"../results/cloth/verts_bending_wi0.001_poking_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release187_Frames_1_increment/"}'\
#        --bending_Subspace_dim 4
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_bend" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"bend":"../results/cloth/verts_bending_wi0.001_poking_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release187_Frames_1_increment/"}'\
#        --bending_Subspace_dim 8
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_bend" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"bend":"../results/cloth/verts_bending_wi0.001_poking_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release187_Frames_1_increment/"}'\
#        --bending_Subspace_dim 9


#python main.py --mesh "cloth" --gravity "active" --example "cloth_bend" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"bend":"../results/cloth/verts_bending_wi0.001_free_falling_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release44_Frames_1_increment/"}'\
#        --bending_Subspace_dim 1
#
#
#python main.py --mesh "cloth" --gravity "active" --example "cloth_bend" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"bend":"../results/cloth/verts_bending_wi0.001_free_falling_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release44_Frames_1_increment/"}'\
#        --bending_Subspace_dim 10

#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_spring" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"spring":"../results/cloth/edge_spring_wi10000.0_stretching_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release40_Frames_1_increment/"}'\
#        --spring_Subspace_dim 2
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_spring" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"spring":"../results/cloth/edge_spring_wi10000.0_stretching_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release40_Frames_1_increment/"}'\
#        --spring_Subspace_dim 10
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_spring" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"spring":"../results/cloth/edge_spring_wi10000.0_stretching_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release40_Frames_1_increment/"}'\
#        --spring_Subspace_dim 20
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_spring" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"spring":"../results/cloth/edge_spring_wi10000.0_stretching_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release40_Frames_1_increment/"}'\
#        --spring_Subspace_dim 40
#
#
#python main.py --mesh "cloth" --gravity "active" --example "cloth_spring" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"spring":"../results/cloth/edge_spring_wi1000.0_twisting_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release45_Frames_1_increment/"}'\
#        --spring_Subspace_dim 6
#
#python main.py --mesh "cloth" --gravity "active" --example "cloth_spring" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"spring":"../results/cloth/edge_spring_wi1000.0_twisting_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release45_Frames_1_increment/"}'\
#        --spring_Subspace_dim 10
#
#python main.py --mesh "cloth" --gravity "active" --example "cloth_spring" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"spring":"../results/cloth/edge_spring_wi1000.0_twisting_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release45_Frames_1_increment/"}'\
#        --spring_Subspace_dim 20

#python main.py --mesh "cloth" --gravity "active" --example "cloth_strain" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"tri_strain":"../results/cloth/tris_strain_wi10.0_stretching_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release40_Frames_1_increment/"}'\
#        --tri_strain_Subspace_dim 6
#
#python main.py --mesh "cloth" --gravity "active" --example "cloth_strain" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"tri_strain":"../results/cloth/tris_strain_wi10.0_stretching_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release40_Frames_1_increment/"}'\
#        --tri_strain_Subspace_dim 10
#
#python main.py --mesh "cloth" --gravity "active" --example "cloth_strain" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"tri_strain":"../results/cloth/tris_strain_wi10.0_stretching_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release40_Frames_1_increment/"}'\
#        --tri_strain_Subspace_dim 20
#
#python main.py --mesh "cloth" --gravity "active" --example "cloth_strain" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"tri_strain":"../results/cloth/tris_strain_wi10.0_stretching_gravity_active_True/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release40_Frames_1_increment/"}'\
#        --tri_strain_Subspace_dim 40

#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_strain" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"tri_strain":"../results/cloth/tris_strain_wi0.02_poking_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release187_Frames_1_increment/"}'\
#        --tri_strain_Subspace_dim 6
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_strain" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"tri_strain":"../results/cloth/tris_strain_wi0.02_poking_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release187_Frames_1_increment/"}'\
#        --tri_strain_Subspace_dim 10
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_strain" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"tri_strain":"../results/cloth/tris_strain_wi0.02_poking_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release187_Frames_1_increment/"}'\
#        --tri_strain_Subspace_dim 20
#
#python main.py --mesh "cloth" --gravity "inactive" --example "cloth_strain" --positionSubspace "none"\
#        --constraintProjectionSubspace "deim_pod_vectorized" \
#        --reducedConstraintProjections '{"tri_strain":"../results/cloth/tris_strain_wi0.02_poking_gravity_active_False/p_bases/deim_pod_vectorized_noAlignement_Volkwein_Standarized_Global_Orthogonalized_Release187_Frames_1_increment/"}'\
#        --tri_strain_Subspace_dim 40

# ----------------------------------------------------------------------------------------------------------------------
## ----------------------------------- LBS for constraint projections ------------------------------------------------##

python main.py --mesh "cloth" --gravity "active" --example "cloth_spring" --positionSubspace "none"\
        --constraintProjectionSubspace "LBS" \
        --reducedConstraintProjections '{"spring":""}'\
        --bending_Subspace_dim 10 --bending_LBSinterpolation_num_samples 100

python main.py --mesh "cloth" --gravity "active" --example "cloth_spring" --positionSubspace "none"\
        --constraintProjectionSubspace "LBS" \
        --reducedConstraintProjections '{"spring":""}'\
        --bending_Subspace_dim 20 --bending_LBSinterpolation_num_samples 200

python main.py --mesh "cloth" --gravity "active" --example "cloth_spring" --positionSubspace "none"\
        --constraintProjectionSubspace "LBS" \
        --reducedConstraintProjections '{"spring":""}'\
        --bending_Subspace_dim 40 --bending_LBSinterpolation_num_samples 400


python main.py --mesh "cloth" --gravity "active" --example "cloth_strain" --positionSubspace "none"\
        --constraintProjectionSubspace "LBS" \
        --reducedConstraintProjections '{"tri_strain":""}'\
        --bending_Subspace_dim 10 --bending_LBSinterpolation_num_samples 100

python main.py --mesh "cloth" --gravity "active" --example "cloth_strain" --positionSubspace "none"\
        --constraintProjectionSubspace "LBS" \
        --reducedConstraintProjections '{"tri_strain":""}'\
        --bending_Subspace_dim 20 --bending_LBSinterpolation_num_samples 200

python main.py --mesh "cloth" --gravity "active" --example "cloth_strain" --positionSubspace "none"\
        --constraintProjectionSubspace "LBS" \
        --reducedConstraintProjections '{"tri_strain":""}'\
        --bending_Subspace_dim 40 --bending_LBSinterpolation_num_samples 400