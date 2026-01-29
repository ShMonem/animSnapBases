import polyscope as ps
import config
import argparse
from config import Config_parameters
import cProfile
import pstats

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
        args.vert_bending_constraint = True
        args.experiments_labels = [6]   # free fall

    elif case == "cloth_spring":
        args.positions_reduced = True
        args.edge_constraint = True
        args.experiments_labels = [2, 3]  # twisting, stretching

    elif case == "cloth_strain":
        args.tri_strain_constraint = True
        args.experiments_labels = [5]  # poking
        args.is_gravity_active = False
        args.strain_limit_constraint_wi =0.02

    elif case == "cloth_bend_spring_strain":
        args.vert_bending_constraint = True
        args.edge_constraint = True
        args.tri_strain_constraint = True

    elif case == "bar_deformationgradient":
        args.tet_deformation_constraint = True

    elif case == "bar_tetstrain":
        args.tet_strain_constraint = True

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

    parser.add_argument('--mesh', type=str, default="cloth", help='Pick a character mesh, available: "cloth", "bar"')
    parser.add_argument('--gravity', type=str, default="active",
                        help='State of gravity, available: "active", "inactive"')
    parser.add_argument('--positionSubspace', type=str, default="none",
                        help='Subspace for positions reduction: "LBS","none"')
    parser.add_argument('--constraintProjectionSubspace', type=str, default="none",
                        help='Subspace for constraint projections reduction: "LBS","none"')

    parser.add_argument('--example', type=str, default="cloth_bend_spring_strain",
                        help='Example settings, available: "cloth_bend", '
                             '"cloth_spring",'
                             ' "cloth_strain", '
                             '"cloth_bend_spring_strain",'
                             '"bar_deformationgradient", '
                             '"bar_tetstrain"')

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
    debug = False

    args.is_gravity_active = (args.gravity == "active")

    args.positions_reduced = (args.positionSubspace != "none")
    if args.positions_reduced: args.position_basis_type = args.positionSubspace

    # args.positions_reduced = (args.positionSubspace != "none")
    # if args.positions_reduced: args.position_basis_type = args.positionSubspace
    #
    example = args.example

    if example == "cloth_bend_spring_strain":
        args.experiments_labels = [0, 1, 2, 3] if args.is_gravity_active else [2, 3, 5, 7]

    object_name = args.mesh

    record_projection_data = True # args.record_projection_data

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