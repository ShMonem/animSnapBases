import polyscope as ps

import config
import argparse


def main(args, record_fom_info = False, case=None, params=None):

    import demos.calbacks
    if case == "testing":
        # callback = demos.calbacks.interacrive_testing_callback(args, record_fom_info, params)
        callback = demos.calbacks.cloth_snapshots(args, record_fom_info, params)
    elif case == "cloth_automated_bend_spring_strain":
        callback = demos.calbacks.cloth_automated_bend_spring_strain_callback(args, record_fom_info, params)
    elif case == "cloth_automated_spring":
        callback = demos.calbacks.cloth_automated_bend_spring_strain_callback(args, record_fom_info, params,
                                                                              experiment="cloth_automated_spring")
    elif case == "cloth_automated_strain":
        callback = demos.calbacks.cloth_automated_strain_callback(args, record_fom_info, params)
    elif case == "cloth_automated_bend":
        callback = demos.calbacks.cloth_automated_bend_callback(args, record_fom_info, params)
    elif case == "bar_automated_deformationgradient":
        callback = demos.calbacks.bar_automated_deformationgradient_callback(args, record_fom_info, params)

    else:
        callback = None
        raise ValueError("callback not set to a true value!")


    # Register callback
    ps.init()
    ps.set_user_callback(callback)

    # Launch viewer
    ps.show()


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------------------------------------
    # available demos:
    #["cloth_automated_bend_spring_strain.json",
    # "cloth_automated_bend.json",
    # "cloth_automated_spring.json",
    # "cloth_automated_strain.json",
    # "bar_automated_deformationgradient.json"]

    # # ---------------- build parser argument ----------------
    parser = argparse.ArgumentParser()

    # Build the system object args holder
    config.initiate_system_args(parser)
    from config import Config_parameters

    param = Config_parameters()
    example = "testing"

    param.reset_parameters("demos/"+example+".json")

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

    record_projection_data = False #args.record_projection_data
    main(args,
         record_fom_info = record_projection_data,
         case = example,
         params = param)