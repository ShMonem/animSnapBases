import polyscope as ps
import config
import argparse


def main(args, record_fom_info = False, case=None, params=None):

    import demos.calbacks
    if case == "testing":
        callback = demos.calbacks.interacrive_testing_callback(args, record_fom_info, params)

    elif case == "cloth_automated_bend_spring_strain":
        callback = demos.calbacks.cloth_automated_bend_spring_strain_callback(args, record_fom_info, params)

    available_demos = {"testing": demos.calbacks.interacrive_testing_callback,
                       "cloth_automated_bend_spring_strain": demos.calbacks.cloth_automated_bend_spring_strain_callback}
    assert case in available_demos

    # Register callback
    ps.init()
    ps.set_user_callback(callback)

    # Launch viewer
    ps.show()


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------------------------------------
    # available demos:
    #["cloth_automated_bend_spring_strain.json"]

    # # ---------------- build parser argument ----------------
    parser = argparse.ArgumentParser()

    # Build the system object args holder
    config.initiate_system_args(parser)
    from config import Config_parameters

    param = Config_parameters()
    param.reset_parameters("demos/cloth_automated_bend_spring_strain.json")

    # Add visualization params
    param.add_visualization_args(parser)

    # Config solver
    param.add_solver_args(parser)

    # Physics parameters
    param.add_physics_args(parser)

    # Model reduction parameters
    param.add_constraint_projections_reduction_args(parser)

    # Important output and input directories
    param.add_directories_args(parser)

    args = parser.parse_args()

    record_projection_data = args.record_projection_data
    example = "cloth_automated_bend_spring_strain"
    main(args,
         record_fom_info = record_projection_data,
         case = example,
         params = param)