from zoneinfo import available_timezones

import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import os

from Constraint_projections import DeformableMesh
from Simulators import Solver
from geometry import get_simple_bar_model, get_simple_cloth_model
from usr_interface import MouseDownHandler, MouseMoveHandler, PreDrawHandler

solver = Solver()
fext = None
model = None
output_path = ""
frame= 0
mouse_down_handler, mouse_move_handler, pre_draw_handler = None, None, None

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

    args = parser.parse_args()

    record_fom_info = True
    example = "cloth_automated_bend_spring_strain"
    main(args,
         record_fom_info = record_fom_info,
         case = example,
         params = param)