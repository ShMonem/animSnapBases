import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
# from scipy.sparse.csgraph import shortest_path
# from scipy.spatial import KDTree
# import config
from Constraint_projections import DeformableMesh
from geometry import get_simple_bar_model, get_simple_cloth_model, get_simple_bar_model_with_surface_points_only, compute_lumped_mass_matrix
from usr_interface import MouseDownHandler, MouseMoveHandler, PreDrawHandler, PickingState

from Simulators import animSnapBasesSolver
# import trimesh
# import meshio
from utils import check_dir_exists, read_mesh_file, compute_vertex_normals, read_obj, rotate_mesh_once_x
from scipy.spatial import cKDTree
# from scipy.spatial.transform import Rotation as R
from demos.calback_utils import *
import demos.calback_utils as callb

# declare global variables
model = None
fext = None
solver = None

picking_state = PickingState()
mouse_down_handler = None
mouse_move_handler = None

# Dicts for collecting snapshots
tets_deformation_gradient_p = {}
positions = {}



# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
def set_up_mouse_handler(args, model, fext):
    # Inside your setup before starting polyscope.show()
    global  mouse_down_handler, mouse_move_handler
    mouse_down_handler = MouseDownHandler(
        is_model_ready=lambda: model is not None,
        picking_state=picking_state,
        solver=solver,
        args=args
    )

    mouse_move_handler = MouseMoveHandler(
        is_model_ready=lambda: model is not None,
        picking_state=picking_state,
        model=model,
        fext=fext
    )

def my_mouse_click_callback(button, modifier):
    if mouse_down_handler is not None:
        return mouse_down_handler.handle_click(button, modifier)
    return False

def my_mouse_move_callback(xpos, ypos):
    if mouse_move_handler is not None:
        return mouse_move_handler.handle_mouse_move(xpos, ypos)
    return False

def get_solver_class_from_name(args):
    if args.solver == "animSnapBasesSolver":
        return animSnapBasesSolver(args)
    else:
        raise ValueError("Unknown solver name")

def rescale(V):
    v_mean = np.mean(V, axis=0)
    V -= v_mean
    scale = np.max(V) - np.min(V)
    if scale != 0:
        V /= scale
    return V

def reset_simulation_model(V, F, T, should_rescale=False, params=None, hight=1):
    global model, fext, solver
    if should_rescale:
        V = rescale(V)

    model = DeformableMesh(V, F, T)
    solver.set_model(model)
    model.set_init_hight(hight)
    fext = np.zeros_like(V)

    ps.remove_all_structures()
    ps.register_surface_mesh("model", model.positions,
                             model.faces, enabled=True)
    ps.get_surface_mesh("model").set_selection_mode("vertices_only")  # or "vertex_only"

    # set soft shadows on the ground
    ps.set_ground_plane_mode("shadow_only")  # tile_reflection , shadow_only
    # Set camera to look down from above, along negative Z
    ps.look_at(
        target=(0.0, 0.0, 0.0),  # Look at the origin (the floor)
        camera_location=(0.0, 0.0, 3.0)  # Camera is 3 units above, looking down
    )

def load_mesh_file(file_name, tetrahedralized=True):
    if tetrahedralized:
        V, T, F = read_mesh_file(file_name)
        return V, T, F
    else:
        V, F = read_obj(file_name)
        return V, None, F
# ----------------------------------------------------------------------------------------------------------------------
# Example callbacks called in main.py
def automated_callback(args, record_fom_info = False,
                                               object_name = None,
                                               tetrahedralized=False,
                                               object_mesh_file ="../data/cloth.obj",
                                               experiment=None,
                                               ):
    experiment = object_name + "_" + experiment
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = args.is_simulating
    output_dir = args.output_dir

    callb.set_automated_experiments(object_name, args)
    solver.set_max_recorded_frames(callb.total_frames)

    def callback():
        nonlocal output_dir, is_simulating

        def reset_record_dir(experiment):
            nonlocal output_dir
            global solver
            # Set main directory
            if record_fom_info and model.record_directory_has_changed:
                output_dir = callb.make_sim_path(args.output_dir, solver, args, object_name, experiment, record_fom_info)
                solver.recording_path = output_dir
                check_dir_exists(solver.recording_path)

                solver.record_path_has_changed = True

                # record parameters for tracking
                with open(solver.recording_path + "/args.txt", "w") as f:
                    for key, value in vars(args).items():
                        f.write(f"{key}: {value}\n")
                model.record_directory_has_changed = False

        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        # Frame 0: create mesh and apply initial constraints
        if solver.frame == 0:
            print(f"Frame {solver.frame}: Creating cloth and fixing left/right corners")

            V, T, F = load_mesh_file(file_name=object_mesh_file, tetrahedralized=tetrahedralized)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)
            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()



            solver.set_dirty()
        # Beginning of Experiments -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Holding/Releasing Sides
        if callb.run_holding_releasing_sides and solver.frame == callb.holding_sides_start_frame:

            reset_record_dir("holding_releasing_sides")

            print(f"Frame {solver.frame}: Start hanging fames")
            model.fix_surface_side_vertices(args.positional_constraint_wi, side="left")
            model.fix_surface_side_vertices(args.positional_constraint_wi, side="right")

        elif callb.run_holding_releasing_sides and solver.frame == callb.release_left_side_frame:
            print(f"Frame {solver.frame}: Releasing right side")
            model.release_surface_side_vertices(side="right")

        elif callb.run_holding_releasing_sides and solver.frame == callb.release_right_side_frame:
            print(f"Frame {solver.frame}: Releasing right left")
            model.release_surface_side_vertices(side="left")

        elif callb.run_holding_releasing_sides and solver.frame == callb.holding_sides_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True
                print(f"Frame {solver.frame}: Storing fames")
        # End of Holding/Releasing Sides Experiments -------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Pinning
        elif callb.run_pinning and solver.frame == callb.pinning_corners_start_frame:

            print(f"Frame {solver.frame}: Start Pinning fames")

            V, T, F = load_mesh_file(file_name=object_mesh_file, tetrahedralized=tetrahedralized)

            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)

            callb.first_fix_frame_pinning(model, object_name, args)
            reset_record_dir("pinning")

        elif callb.run_pinning and solver.frame == callb.release_pinning_left_corner_frame:
            callb.first_release_frame_pinning(model, object_name, solver.frame)

        elif callb.run_pinning and solver.frame == callb.release_pinning_right_corner_frame:
            callb.second_release_frame_pinning(model, object_name, solver.frame)

        elif callb.run_pinning and solver.frame == callb.pinning_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True
                print(f"Frame {solver.frame}: Storing fames")

        # End of Pinning Experiments ----------------------------------------------------------------.------------------
        # --------------------------------------------------------------------------------------------------------------
        # Twisting
        elif callb.run_twisting and solver.frame == callb.twisting_start_frame:

            print(f"Frame {solver.frame}: Start twisting fames")

            V, T, F = load_mesh_file(file_name=object_mesh_file, tetrahedralized=tetrahedralized)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)

            solver.reference_frame = solver.frame

            callb.first_fix_pick__frame_twisting(model, object_name, args, solver.reference_frame)
            reset_record_dir("twisting")

        elif callb.run_twisting and solver.frame == callb.release_twisting_start_frame:
            print(f"Frame {solver.frame}: Releasing left side")
            callb.first_release_frame_twisting(model, object_name, solver.frame)
            solver.set_dirty()

        elif callb.run_twisting and solver.frame == callb.twisting_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True

        # End of Twisting Experiments ----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Stretching
        elif callb.run_stretching and solver.frame == callb.stretching_start_frame:

            print(f"Frame {solver.frame}: Start stretching fames")

            V, T, F = load_mesh_file(file_name=object_mesh_file, tetrahedralized=tetrahedralized)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)

            solver.reference_frame = solver.frame

            model.compute_sides_and_corner_indices()

            callb.first_fix_pick_frame_stretching(model, object_name, args, solver.frame)
            reset_record_dir("stretching")

            solver.set_dirty()
            print("Stretching - positional constraint added to right and left sides.")

        elif callb.run_stretching and solver.frame == callb.release_stretching_start_frame:

            callb.first_release_frame_stretshing(model, object_name, solver.frame)
            solver.set_dirty()

        elif callb.run_stretching and solver.frame == callb.stretching_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True
        # End of Stretching Experiments --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Squeezing
        elif callb.run_squeezing and solver.frame == callb.squeezing_start_frame:

            print(f"Frame {solver.frame}: Start squeezing fames")

            V, T, F = load_mesh_file(file_name=object_mesh_file, tetrahedralized=tetrahedralized)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)

            solver.reference_frame = solver.frame

            model.compute_sides_and_corner_indices()
            callb.first_fix_pick__frame_squeezing(model, object_name, args, solver.frame)
            reset_record_dir("squeezing")

            solver.set_dirty()
            print("Squeezing - positional constraint added to right and left sides.")

        elif callb.run_squeezing and solver.frame == callb.squeezing_end_frame:
            callb.first_release_frame_squeezing(model, object_name, solver.frame)
            solver.set_dirty()
            if record_fom_info:
                solver.store_current_snapshots = True

        # End of Squeezing Experiments ---------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Poking
        elif callb.run_poking and solver.frame == callb.poking_start_frame:

            print(f"Frame {solver.frame}: Start poking frames")

            V, T, F = load_mesh_file(file_name=object_mesh_file, tetrahedralized=tetrahedralized)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)

            solver.reference_frame = solver.frame

            callb.poked_points, callb.labels = callb.compute_voronoi_seeds_incremental(model.init_positions, callb.number_poking_points, visualize=False)

            callb.poking_motion, _, _ = callb.create_poking_motions_at_given_seeds(model.positions,
                                                                        callb.poked_points,
                                                                        direction="z",     # "normal" or "x"/"y"/"z"
                                                                        F=model.faces,              # required if direction="normal"
                                                                        f_l=callb.number_frames_per_poke, # frames for motion phase
                                                                        f_j=callb.number_frames_rest_per_poke,  # frames for rest phase
                                                                        amplitude=callb.poking_amplitude,          # displacement magnitude (same units as V)
                                                                        repeats=1,              # how many poke cycles per seed (when sequential)
                                                                        mode="sequential",      # "sequential" or "simultaneous"
                                                                        normalize_dir=True,     # normalize direction vectors
                                                                    )
            model.add_positional_constraint(callb.poked_points[0], args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=callb.poking_motion[0], frame_reset=solver.frame)
            print("Poking - positional constraint added to first vertex")

            model.picked_vert[callb.poked_points[0]] = True
            callb.poking_count +=1

            reset_record_dir("poking")

        elif (callb.run_poking and callb.poking_end_frame > solver.frame > callb.poking_start_frame
                and (solver.frame - callb.poking_start_frame) % (callb.number_frames_per_poke+callb.number_frames_rest_per_poke) == 0) :

            label = (solver.frame - callb.poking_start_frame) // (callb.number_frames_per_poke+callb.number_frames_rest_per_poke)
            model.remove_positional_constraint(callb.poked_points[label-1])
            solver.set_dirty()

            if solver.frame < callb.poking_end_frame and callb.poking_count < callb.number_poking_points:
                print(f"Poking - positional constraint added to {label+1}th vertex")
                model.add_positional_constraint(callb.poked_points[label], args.positional_constraint_wi,
                                                motion_type="user_defined", frames_series=callb.poking_motion[label],
                                                frame_reset=solver.frame)
                model.picked_vert[callb.poked_points[label]] = True
                callb.poking_count += 1

        elif callb.run_poking and solver.frame == callb.poking_end_frame:
            # model.remove_positional_constraint(callb.poked_points[0])
            if record_fom_info:
                solver.store_current_snapshots = True
        # End of Poking Experiments ------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Vertical Fall
        elif callb.run_falling and solver.frame == callb.gravitational_fall_start_frame:

            print(f"Frame {solver.frame}: Starting free fall frames")

            V, T, F = load_mesh_file(file_name=object_mesh_file, tetrahedralized=tetrahedralized)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)
            solver.reference_frame = solver.frame

            reset_record_dir("free_falling")

        elif callb.run_falling and solver.frame == callb.gravitational_fall_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True
        # End of Vertical Falling Experiments --------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Axial Rotation
        elif callb.run_rotating and solver.frame == callb.rotating_start_frame:

            print(f"Frame {solver.frame}: Starting rotating frames")

            V, T, F = load_mesh_file(file_name=object_mesh_file, tetrahedralized=tetrahedralized)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)
            solver.reference_frame = solver.frame

            callb.create_full_rotating_motion(model.init_positions.copy(), callb.number_rotating_frames)

            reset_record_dir("rotating")

        elif callb.run_rotating and callb.rotating_end_frame > solver.frame > callb.rotating_start_frame:
            f = solver.frame - solver.reference_frame
            model.positions[:] = callb.rotating_positions_series[f]

        elif callb.run_rotating and solver.frame == callb.rotating_end_frame:
            print(f"Frame {solver.frame}: Ending rotating frames")

            if record_fom_info:
                solver.store_current_snapshots = True

        # End of Axial Rotation Experiments ----------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # End of Experiments -------------------------------------------------------------------------------------------

        elif solver.frame == callb.total_frames:
            print(f"Frame {solver.frame}: End of experiments")

            print("Stopping simulation.")
            is_simulating = False
            ps.unshow()
            return

        # Run a single simulation step
        if model is not None and is_simulating:


            pre_draw_handler = PreDrawHandler(
                lambda: model.positions.shape[0] > 0, args, solver, fext,
                record_info=record_fom_info, record_path=solver.recording_path
            )
            pre_draw_handler.set_animating(True)
            pre_draw_handler.handle()

        if model is not None:
            psim.BulletText(f"Vertices: {model.positions.shape[0]}")
            psim.BulletText(f"Triangles: {model.faces.shape[0]}")
            psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
            # psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

            if model.has_verts_bending_constraints:
                psim.BulletText(f"Vertices bending constraint: {len(model.verts_bending_constraints)}")
                psim.BulletText(f"wi: { str(args.vert_bending_constraint_wi) }")

            if model.has_edge_spring_constraints:
                psim.BulletText(f"Edge pring constraint: {len(model.edge_spring_constraints)}")
                psim.BulletText(f"wi: { str(args.edge_constraint_wi) }")

            if model.has_tris_strain_constraints:
                psim.BulletText(f"Triangles strain constraint: {len(model.tris_strain_constraints)}")
                psim.BulletText(f"wi: { str(args.strain_limit_constraint_wi) }")

        psim.End()
    return callback




#
# def interacrive_testing_callback(args, record_fom_info = False, params=None, experiment="testing"):
#     global model, fext, solver, mouse_down_handler, mouse_move_handler
#     solver = get_solver_class_from_name(args)
#     is_simulating = args.is_simulating
#     output_path = args.output_dir
#
#     def callback():
#         nonlocal output_path
#         psim.PushItemWidth(200)
#         psim.TextUnformatted("== Projective Dynamics ==")
#         psim.Separator()
#         object_name = ""
#
#         record = False
#         system_name = "User_defined"
#
#         def make_sim_path(args):
#             nonlocal output_path
#
#             sim_case = "FOM"
#
#             if solver.has_reduced_position and not solver.has_reduced_constraint_projections:
#                 sim_case = "positions_reduced/" + args.position_basis_type
#             elif solver.has_reduced_constraint_projections and not solver.has_reduced_position:
#                 sim_case = "constraint_projections_reduced/" + args.constraint_projection_basis_type
#
#             elif solver.has_reduced_constraint_projections and solver.has_reduced_position:
#                 sim_case = "positions_and_constraint_projections_reduced/" + args.position_basis_type + "_" + args.constraint_projection_basis_type
#
#             specify_path = ""
#             if model.has_verts_bending_constraints:
#                 specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
#                 if args.vert_bending_reduced:
#                     specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"
#
#             if model.has_edge_spring_constraints:
#                 specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
#                 if args.edge_spring_reduced:
#                     specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"
#
#             if model.has_tris_strain_constraints:
#                 specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
#                 if args.tri_strain_reduced:
#                     specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
#             if model.has_tets_strain_constraints:
#                 specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
#                 if args.tet_strain_reduced:
#                     specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
#             if model.has_tets_deformation_gradient_constraints:
#                 specify_path = specify_path + "tets_deformation_gradient_wi" + str(
#                     args.deformation_gradient_constraint_wi) + "_"
#                 if args.tet_deformation_reduced:
#                     specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components) + "_"
#             output_path += "/" + object_name + "/" + experiment + "/" + sim_case + "/" + specify_path + "/"
#             check_dir_exists(output_path)
#
#             solver.set_record_path(output_path)
#             solver.set_store_p(record_fom_info)
#             solver.set_store_q(record_fom_info)
#
#         if solver.frame == 0:
#             psim.PushItemWidth(200)
#             psim.TextUnformatted("== Projective Dynamics ==")
#             psim.Separator()
#
#             if system_name == "Bar":
#
#                 V, T, F = read_mesh_file("../data/bar.mesh")
#                 reset_simulation_model(V, F, T, should_rescale=True)
#                 object_name = "bar"
#
#                 if record_fom_info:
#                     make_sim_path(args)
#                     # record parameters for tracking
#                     with open(output_path + "/args.txt", "w") as f:
#                         for key, value in vars(args).items():
#                             f.write(f"{key}: {value}\n")
#
#             if system_name == "Cloth":
#
#                 V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
#                 reset_simulation_model(V, F, None, should_rescale=True)
#                 object_name = "Cloth"
#
#                 if record_fom_info:
#                     make_sim_path(args)
#                     # record parameters for tracking
#                     with open(output_path + "/args.txt", "w") as f:
#                         for key, value in vars(args).items():
#                             f.write(f"{key}: {value}\n")
#
#             if system_name == "User_defined":
#                 V, T, F = read_mesh_file("../data/sphere.mesh")
#                 reset_simulation_model(V, F, T, should_rescale=True)
#                 object_name = "sphere"
#
#                 if record_fom_info:
#                     make_sim_path(args)
#                     # record parameters for tracking
#                     with open(output_path + "/args.txt", "w") as f:
#                         for key, value in vars(args).items():
#                             f.write(f"{key}: {value}\n")
#                 model.fix_surface_side_vertices(args.positional_constraint_wi, side="left")
#
#             solver.set_dirty()
#
#         if model is not None:
#             set_up_mouse_handler(args, model, fext)
#             psim.BulletText(f"Vertices: {model.positions.shape[0]}")
#             psim.BulletText(f"Triangles: {model.faces.shape[0]}")
#             psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
#             psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")
#
#         if model is not None and is_simulating:
#
#             pre_draw_handler = PreDrawHandler(
#                 lambda: model.positions.shape[0] > 0, args, solver, fext,
#                 record_info=record_fom_info, record_path=output_path
#             )
#             pre_draw_handler.set_animating(True)
#             pre_draw_handler.handle()
#         # if psim.CollapsingHeader("Physics"):
#         #     if psim.TreeNode("Constraints"):
#         #
#         #         if object_name == "Bar":
#         #             changed, args.fix_left_side = psim.Checkbox("Fix Left\nVertices Side", args.fix_left_side)
#         #             changed, args.fix_right_side = psim.Checkbox("Fix Right\nVertices Side", args.fix_right_side)
#         #
#         #         if object_name == "Cloth":
#         #             changed, args.fix_left_corners = psim.Checkbox("Fix Left\nCorners Side", args.fix_left_corners)
#         #             changed, args.fix_right_corners = psim.Checkbox("Fix Right\nCorners Side", args.fix_right_corners)
#         #
#         #             changed, args.fix_top_corners = psim.Checkbox("Fix Top\nCorners Side", args.fix_top_corners)
#         #             changed, args.fix_bottom_corners = psim.Checkbox("Fix Bottom\nCorners Side",
#         #                                                              args.fix_bottom_corners)
#         #
#         #         changed, args.vert_bending_constraint_wi = psim.InputFloat("wi \nVertBend",
#         #                                                                    args.vert_bending_constraint_wi)
#         #         changed, args.vert_bending_constraint = psim.Checkbox("Active \nVertBend", args.vert_bending_constraint)
#         #
#         #         changed, args.edge_constraint_wi = psim.InputFloat("wi \nEdgeSpring", args.edge_constraint_wi)
#         #         changed, args.edge_constraint = psim.Checkbox("Active \nEdgeSpring", args.edge_constraint)
#         #
#         #         changed, args.deformation_gradient_constraint_wi = psim.InputFloat("wi \nDeformationGradient",
#         #                                                                            args.deformation_gradient_constraint_wi)
#         #         changed, args.tet_deformation_constraint = psim.Checkbox("Active \nDeformationGradient",
#         #                                                                  args.tet_deformation_constraint)
#         #
#         #         changed, args.strain_limit_constraint_wi = psim.InputFloat("wi \nStrainLimit",
#         #                                                                    args.strain_limit_constraint_wi)
#         #         changed, args.sigma_min = psim.InputFloat("Minimum singular \nvalue StrainLimit", args.sigma_min)
#         #         changed, args.sigma_max = psim.InputFloat("Maximum singular \nvalue StrainLimit", args.sigma_max)
#         #
#         #         changed, args.tri_strain_constraint = psim.Checkbox("Active \nTriStrain", args.tri_strain_constraint)
#         #         changed, args.tet_strain_constraint = psim.Checkbox("Active \nTetStrain", args.tet_strain_constraint)
#         #
#         #         changed, args.positional_constraint_wi = psim.InputFloat("wi \nPositional constraint",
#         #                                                                  args.positional_constraint_wi)
#         #
#         #         if psim.Button("Apply##Constraints"):
#         #             model.immobilize()
#         #             model.clear_constraints()
#         #             # model.reset_constraints_attributes()
#         #             solver.set_dirty()
#         #             # ---------------------------------------------------------------------------------------------------
#         #
#         #             # used for Bar
#         #             if args.fix_left_side and not args._fix_left_triggered:
#         #                 model.fix_surface_side_vertices(side="left")
#         #                 args._fix_left_triggered = True
#         #             elif args._fix_left_triggered and not args.fix_left_side:
#         #                 model.release_surface_side_vertices(side="left")
#         #                 args._fix_left_triggered = False
#         #
#         #             if args.fix_right_side and not args._fix_right_triggered:
#         #                 model.fix_surface_side_vertices(side="right")
#         #                 args._fix_right_triggered = True
#         #             elif args._fix_right_triggered and not args.fix_right_side:
#         #                 model.release_surface_side_vertices(side="right")
#         #                 args._fix_right_triggered = False
#         #             # ---------------------------------------------------------------------------------------------------
#         #
#         #             # used for cloth
#         #             if args.fix_top_corners and not args._fix_top_corners_triggered:
#         #                 model.fix_cloth_corners(side="top")
#         #                 args._fix_top_corners_triggered = True
#         #             elif args._fix_top_corners_triggered and not args.fix_top_corners:
#         #                 model.release_cloth_corners(side="top")
#         #                 args._fix_top_corners_triggered = False
#         #
#         #             if args.fix_bottom_corners and not args._fix_bottom_corners_triggered:
#         #                 model.fix_cloth_corners(side="bottom")
#         #                 args._fix_bottom_corners_triggered = True
#         #             elif args._fix_bottom_corners_triggered and not args.fix_bottom_corners:
#         #                 model.release_cloth_corners(side="bottom")
#         #                 args._fix_bottom_corners_triggered = False
#         #
#         #             if args.fix_right_corners and not args._fix_right_corners_triggered:
#         #                 model.fix_cloth_corners(side="right")
#         #                 args._fix_right_corners_triggered = True
#         #             elif args._fix_right_corners_triggered and not args.fix_right_corners:
#         #                 model.release_cloth_corners(side="right")
#         #                 args._fix_right_corners_triggered = False
#         #
#         #             if args.fix_left_corners and not args._fix_left_corners_triggered:
#         #                 model.fix_cloth_corners(side="left")
#         #                 args._fix_left_corners_triggered = True
#         #             elif args._fix_left_corners_triggered and not args.fix_left_corners:
#         #                 model.release_cloth_corners(side="left")
#         #                 args._fix_left_corners_triggered = False
#         #             # ---------------------------------------------------------------------------------------------------
#         #
#         #         psim.BulletText(f"no. Constraints: {len(model.constraints)}")
#         #         psim.TreePop()
#         #
#         #     changed, args.dt = psim.InputFloat("Timestep", args.dt)
#         #     changed, args.solver_iterations = psim.InputInt("Solver iterations", args.solver_iterations)
#         #     changed, args.mass_per_particle = psim.InputFloat("mass per particle", args.mass_per_particle)
#         #     changed, args.is_gravity_active = psim.Checkbox("Gravity", args.is_gravity_active)
#         #
#         #     changed, args.is_simulating = psim.Checkbox("Simulate", args.is_simulating)
#         #
#         #
#         #
#         #     if model is not None:
#         #
#         #         # # if recording snapshots build output file name/ path
#         #         # if record_fom_info:
#         #         #     specify_path = ""
#         #         #     if model.has_verts_bending_constraints:
#         #         #         specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
#         #         #         if args.vert_bending_reduced:
#         #         #             specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"
#         #         #
#         #         #     if model.has_edge_spring_constraints:
#         #         #         specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
#         #         #         if args.edge_spring_reduced:
#         #         #             specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"
#         #         #
#         #         #     if model.has_tris_strain_constraints:
#         #         #         specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
#         #         #         if args.tri_strain_reduced:
#         #         #             specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
#         #         #     if model.has_tets_strain_constraints:
#         #         #         specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
#         #         #         if args.tet_strain_reduced:
#         #         #             specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
#         #         #     if model.has_tets_deformation_gradient_constraints:
#         #         #         specify_path = specify_path + "tets_deformation_gradient_wi" + str(
#         #         #             args.deformation_gradient_constraint_wi) + "_"
#         #         #         if args.tet_deformation_reduced:
#         #         #             specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components)+ "_"
#         #         #
#         #         #     output_path += "/" + object_name + "/" + specify_path
#         #
#         #         # mouse_down_handler = MouseDownHandler(lambda: model.positions.shape[0] > 0, picking_state, solver, physics_params)
#         #         # mouse_move_handler = MouseMoveHandler(lambda: model.positions.shape[0] > 0, picking_state, model, lambda: fext)
#         #         fext_dragging = mouse_move_handler.fext
#         #
#         #         pre_draw_handler = PreDrawHandler(lambda: model.positions.shape[0] > 0, args, solver, fext ,
#         #                                           record_info=record_fom_info, record_path=output_path)
#         #         # print(solver.frame)
#         #     if args.is_simulating:
#         #         pre_draw_handler.set_animating(True)
#         #         pre_draw_handler.handle()
#
#         # Inside interactive_testing_callback
#         # io = psim.GetIO()
#         # if io.MouseClicked[0]:  # left-click
#         #     screen_coords = io.MousePos
#         #     current_x, current_y = screen_coords
#         #     pick_result = ps.pick(screen_coords=screen_coords)
#         #
#         #     if pick_result.is_hit and pick_result.structure_name == "model":
#         #         # Get modifier
#         #         if io.KeyCtrl:
#         #             # dragging mode
#         #             modifier = "ctrl"
#         #         elif io.KeyShift:
#         #             # add positional constraint
#         #             modifier = "shift"
#         #
#         #         else:
#         #             modifier = None
#         #
#         #         v_id = pick_result.local_index
#         #         pos = pick_result.position
#         #
#         #         picking_state.vertex = v_id
#         #         picking_state.is_picking = (modifier == "ctrl")
#         #         picking_state.mouse_x = current_x
#         #         picking_state.mouse_y = current_y
#         #         print(f"Picked vertex {v_id} at screen {screen_coords} -> position {pos} --> modifier {modifier}")
#         #
#         #         mouse_down_handler.handle_click(pick_result, button="left", modifier=modifier)
#         #     if picking_state.is_picking and mouse_move_handler is not None:
#         #
#         #         mouse_move_handler.handle_mouse_move()
#         #
#         # if psim.Button("Cancel Picking"):
#         #     picking_state.is_picking = False
#         #     model.picked_vert = [False] *len(model.picked_vert )
#         #
#         # if psim.CollapsingHeader("Visualization"):
#         #     changed, wire = psim.Checkbox("Wireframe", ps.get_surface_mesh("mesh").get_edge_width() > 0.0)
#         #     if wire:
#         #         ps.get_surface_mesh("mesh").set_edge_width(1.0)
#         #     else:
#         #         ps.get_surface_mesh("mesh").set_edge_width(0.0)
#         #     ps.get_surface_mesh("mesh").set_point_radius(psim.InputFloat("Point size", 0.02), relative=True)
#         #
#         # psim.End()
#         psim.End()
#     return callback
