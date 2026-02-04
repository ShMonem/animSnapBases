import polyscope as ps
import config
import argparse
from config import Config_parameters
import cProfile
import pstats
import json

def main(args, record_fom_info = False, case=None,object_name=None, mesh_file =None, tetrahedralized=False):

    import demos.calbacks
    # Pre-designed experiments -----------------------------------------------------------------------------------------
    # predefined_experiments_in_order = ["holding_releasing_sides",  # 0
    #                                    "pinning_release_sides",  # 1
    #                                    "twisting",  # 2
    #                                    "stretching",  # 3
    #                                    "squeezing",  # 4
    #                                    "poking",  # 5
    #                                    "free_falling",  # 6
    #                                    "rotating",  # 7
    #                                    ]

    # if case == "testing":
    #     callback = demos.calbacks.interacrive_testing_callback(args, record_fom_info, params)
    # ------------------------------------------------------------------------------------------------------------------
    if case == "cloth_bend":
        args.experiments_labels = [6] if args.is_gravity_active else [5]

        args.vert_bending_constraint = True
        # args.experiments_labels = [6]   # free fall

    elif case == "cloth_spring":
        args.edge_constraint = True
        args.pos_radial_r_muliplier = 2.0
        if args.constraint_projection_basis_type == "deim_pod_vectorized":
            args.experiments_labels = [2] if args.is_gravity_active else [3]
        elif args.constraint_projection_basis_type == "LBS":
            args.experiments_labels = [1, 2, 3]
        else:
            args.experiments_labels = [0, 1, 2, 3] if args.is_gravity_active else [2, 3, 5]

    elif case == "cloth_strain":
        args.tri_strain_constraint = True
        if args.constraint_projection_basis_type == "deim_pod_vectorized":
            args.experiments_labels = [3] if args.is_gravity_active else [5]
        else:
            args.experiments_labels = [3] if args.is_gravity_active else [3, 5]
        args.strain_limit_constraint_wi =0.02

    elif case == "cloth_bend_spring_strain":
        args.experiments_labels = [0, 1, 2, 3] if args.is_gravity_active else [2, 3, 5, 7]

        args.vert_bending_constraint = True
        args.edge_constraint = True
        args.tri_strain_constraint = True

    elif case == "bar_deformationgradient":
        args.tet_deformation_constraint = True
        args.pos_radial_r_muliplier = 2.0
        args.experiments_labels = [0, 2] if args.is_gravity_active else [2, 4]

    #
    # elif case == "bar_tetstrain":
    #     args.tet_strain_constraint = True
    #     args.experiments_labels = [0, 1, 6] if args.is_gravity_active else []


    else:
        raise ValueError("Callback not set to a true value!")

    callback = demos.calbacks.automated_callback(args, record_fom_info,
                                                 object_name=object_name, object_mesh_file=mesh_file,
                                                 tetrahedralized=tetrahedralized, experiment=case)

    # Register callback
    ps.init()
    ps.set_user_callback(callback)

    # Launch viewer
    ps.show()


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------------------------------------
    # available demos:
    #[
    # "cloth_bend.json",
    # "cloth_spring",
    # "cloth_strain",
    # "cloth_bend_spring_strain"
    # "bar_tetstrain",
    # "bar_deformationgradient"]

    # # ---------------- build parser argument ----------------
    parser = argparse.ArgumentParser(description="Set base variables/parameters.")


    parser.add_argument('--record', type=bool, default=True,
                        help='Record snapshots and images, available: "True", "False"')
    # mesh character
    parser.add_argument('--mesh', type=str, default="cloth", help='Pick a character mesh, available: "cloth", "bar"')

    # example: constraints
    parser.add_argument('--example', type=str, default="cloth_strain",
                        help='Example settings, available: "cloth_bend", '
                             '"cloth_spring",'
                             ' "cloth_strain", '
                             '"cloth_bend_spring_strain",'
                             '"bar_deformationgradient", '
                             )
    # gravity status
    parser.add_argument('--gravity', type=str, default="active",
                        help='State of gravity, available: "active", "active"')
    # position reduction
    parser.add_argument('--positionSubspace', type=str, default="none",
                        help='Subspace for positions reduction: "LBS","none"')


    # constraint projection reduction parameters
    parser.add_argument('--constraintProjectionSubspace', type=str, default="LBS",
                        help='Subspace for constraint projections reduction: "deim_pod_vectorized", "LBS","none"')


    parser.add_argument('--reducedConstraintProjections', type=json.loads, default={"tri_strain":""},
                        help='Reduced constraint projections, can be a subset of: {"bend": "file"}')
    # bending
    parser.add_argument('--bending_Subspace_dim', type=int, default=100,
                        help='Subspace dim for constraint projections reduction')
    parser.add_argument('--bending_LBSinterpolation_num_samples', type=int, default=100,
                        help='Number of samples for constraint projections reduction')
    # spring
    parser.add_argument('--spring_Subspace_dim', type=int, default=100,
                        help='Subspace dim for constraint projections reduction')
    parser.add_argument('--spring_LBSinterpolation_num_samples', type=int, default=400,
                        help='Number of samples for constraint projections reduction')
    # spring
    parser.add_argument('--tri_strain_Subspace_dim', type=int, default=40,
                        help='Subspace dim for constraint projections reduction')
    parser.add_argument('--tri_strain_LBSinterpolation_num_samples', type=int, default=300,
                        help='Number of samples for constraint projections reduction')




    # Build the system object args holder
    config.initiate_system_args(parser)

    param = Config_parameters()
    # First we parse ONLY what's known
    early_args, _ = parser.parse_known_args()
    mesh_file = None
    tetrahedralized = False

    if early_args.example in {"bar_deformationgradient", "bar_tetstrain"}:
        param.reset_parameters("demos/bar_params.json")
        mesh_file = "../data/bar.mesh"
        tetrahedralized = True
    elif early_args.example in {"cloth_bend", "cloth_spring", "cloth_strain", "cloth_bend_spring_strain"}:
        param.reset_parameters("demos/cloth_params.json")
        mesh_file = "../data/cloth.obj"
        tetrahedralized = False
    else:
        raise ValueError(f"Unknown example: {early_args.example}")

    # Add visualization params
    param.add_visualization_args(parser)

    # Config solver
    param.add_solver_args(parser)

    # Physics parameters
    param.add_physics_args(parser)

    # Model reduction parameters
    # positions
    param.add_position_reduction_args(parser)
    # constraints projections
    param.add_constraint_projections_reduction_args(parser)

    # Important output and input directories
    param.add_directories_args(parser)

    args = parser.parse_args()

    args.is_gravity_active = (args.gravity == "active")


    args.positions_reduced = (args.positionSubspace != "none")
    if args.positions_reduced: args.position_basis_type = args.positionSubspace

    args.constProj_reduced = (args.constraintProjectionSubspace != "none")
    if args.constProj_reduced:
        args.constraint_projection_basis_type = args.constraintProjectionSubspace
        if args.constraint_projection_basis_type in {"deim_pod_vectorized", "LBS"}:
            if "bend" in args.reducedConstraintProjections:
                args.vert_bending_reduced = True
                args.interpolation_basis_dir = args.reducedConstraintProjections["bend"]
                args.vert_bending_num_components = args.bending_Subspace_dim
                args.vert_bending_num_samples = args.bending_LBSinterpolation_num_samples
            if "spring" in args.reducedConstraintProjections:
                args.edge_spring_reduced = True
                args.interpolation_basis_dir = args.reducedConstraintProjections["spring"]
                args.edge_spring_num_components = args.spring_Subspace_dim
                args.edge_spring_num_samples = args.spring_LBSinterpolation_num_samples
            if "tri_strain" in args.reducedConstraintProjections:
                args.tri_strain_reduced = True
                args.interpolation_basis_dir = args.reducedConstraintProjections["tri_strain"]
                args.tri_strain_num_components = args.tri_strain_Subspace_dim
                args.tri_strain_num_samples = args.tri_strain_LBSinterpolation_num_samples




    example = args.example

    object_name = args.mesh

    record_projection_data = args.record # args.record_projection_data

    debug = False
    if debug:
        with cProfile.Profile() as pr:
            main(args,
                 record_fom_info=record_projection_data,
                 case=example,
                 object_name=object_name,
                 mesh_file=mesh_file,
                 tetrahedralized=tetrahedralized)

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME).print_stats(100)
    else:
        main(args,
             record_fom_info=record_projection_data,
             case=example,
             object_name = object_name,
             mesh_file =mesh_file,
             tetrahedralized=tetrahedralized)