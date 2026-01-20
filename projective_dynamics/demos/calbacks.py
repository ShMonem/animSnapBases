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
from utils import check_dir_exists, read_mesh_file, compute_vertex_normals
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

    ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
    # Set camera to look down from above, along negative Z
    ps.look_at(
        target=(0.0, 0.0, 0.0),  # Look at the origin (the floor)
        camera_location=(0.0, 0.0, 3.0)  # Camera is 3 units above, looking down
    )

# ----------------------------------------------------------------------------------------------------------------------
# Example callbacks called in main.py
def bar_automated_callback(args, record_fom_info = False,
                                               params=None,
                                               object = "bar",
                                               experiment="automated_deformationgradient",
                                               ):
    experiment = object + "_" + experiment
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = args.is_simulating
    output_dir = args.output_dir

    callb.set_automated_experiments(object, args)
    if callb.total_frames > args.max_p_snapshots_num:
        solver.set_max_recorded_frames(callb.total_frames)

    def callback():
        nonlocal output_dir, is_simulating
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        # Frame 0: create mesh and apply initial constraints
        if solver.frame == 0:
            V, T, F = read_mesh_file("../data/bar.mesh")

            # params.edit_system_args(args, "Bar")
            # V, T, F, _ = get_simple_bar_model(args.bar_width, args.bar_height, args.bar_depth)

            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)

            object_name = "bar"
            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            if record_fom_info:
                output_dir = callb.make_sim_path(output_dir, solver, args, object_name, experiment, record_fom_info)
                # record parameters for tracking
                with open(output_dir + "/args.txt", "w") as f:
                    for key, value in vars(args).items():
                        f.write(f"{key}: {value}\n")

            solver.set_dirty()

        if callb.run_holding_releasing_sides and solver.frame == callb.holding_sides_start_frame:

            solver.recording_path = os.path.join(output_dir,"holding_releasing_sides")
            solver.record_path_has_changed = True
            check_dir_exists(solver.recording_path )
            print(f"Frame {solver.frame}: Start hanging fames")
            model.fix_surface_side_vertices(args.positional_constraint_wi, side="left")
            model.fix_surface_side_vertices(args.positional_constraint_wi, side="right")

        elif callb.run_holding_releasing_sides and solver.frame == callb.release_left_side_frame:
            print(f"Frame {solver.frame}: Releasing left side")
            model.release_surface_side_vertices(side="left")

        elif callb.run_holding_releasing_sides and solver.frame == callb.release_right_side_frame:
            print(f"Frame {solver.frame}: Releasing right side")
            model.release_surface_side_vertices(side="right")

        elif callb.run_holding_releasing_sides and solver.frame == callb.holding_sides_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True

        elif callb.run_twisting and solver.frame == callb.twisting_start_frame:

            solver.recording_path = os.path.join(output_dir, "twisting")
            solver.record_path_has_changed = True
            check_dir_exists(solver.recording_path)
            print(f"Frame {solver.frame}: Start twisting fames")

            V, T, F = read_mesh_file("../data/bar.mesh")
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)

            model.fix_surface_side_vertices(args.positional_constraint_wi, side="right")
            solver.reference_frame = solver.frame
            side_verts = model.toggle_pick_surface_side_vertices( side="left", return_surface_verts=True) # pick

            motions, axis_yz = callb.create_surface_twist_motions(
                V=V,
                surface_verts=side_verts,
                theta_max=-callb.max_theta,  # 180 degrees
                num_frames=callb.number_twisting_frames,
                axis="x",
                ease="linear",
                hold_frames=10
            )
            for vi in side_verts:
                model.add_positional_constraint(
                    vi,
                    wi=args.positional_constraint_wi,
                    motion_type="user_defined",
                    frames_series=motions[vi],
                    frame_reset=solver.frame
                )
            # solver.set_dirty()

        elif callb.run_twisting and solver.frame == callb.release_twisting_start_frame:
            print(f"Frame {solver.frame}: Releasing left side")
            side_verts = model.toggle_pick_surface_side_vertices( side="left", return_surface_verts=True) # pick

            for vi in side_verts:
                model.remove_positional_constraint(vi)
            solver.set_dirty()

        elif callb.run_twisting and solver.frame == callb.twisting_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True

        elif callb.run_stretching and solver.frame == callb.stretching_start_frame:

            solver.recording_path = os.path.join(output_dir,"stretching")
            solver.record_path_has_changed = True
            check_dir_exists(solver.recording_path)
            print(f"Frame {solver.frame}: Start stretching fames")

            V, T, F = read_mesh_file("../data/bar.mesh")
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)
            solver.reference_frame = solver.frame

            model.compute_sides_and_corner_indices()
            right_side_verts = model._side_surface_verts["right"]
            left_side_verts = model._side_surface_verts["left"]

            # Generate serise for streatching
            stretch_motion_x_axis_right = callb.create_xyz_stretch_motion_with_jumps(callb.number_stretching_frames, 0,
                                                                      1,
                                                                      displacement_xyz=(0.4, 0.0, 0.0))
            stretch_motion_x_axis_left = - stretch_motion_x_axis_right

            for v in right_side_verts:
                model.add_positional_constraint(v, args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=stretch_motion_x_axis_right, frame_reset=solver.frame)
                model.picked_vert[v] = True

            for v in left_side_verts:
                model.add_positional_constraint(v, args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=stretch_motion_x_axis_left, frame_reset=solver.frame)
                model.picked_vert[v] = True

            solver.set_dirty()
            print("Stretching - positional constraint added to right and left sides.")

        elif callb.run_stretching and solver.frame == callb.release_stretching_start_frame:

            print(f"Frame {solver.frame}: Releasing left side")
            side_verts = model.toggle_pick_surface_side_vertices(side="left", return_surface_verts=True)  # pick

            for vi in side_verts:
                model.remove_positional_constraint(vi)
            solver.set_dirty()

        elif callb.run_stretching and solver.frame == callb.stretching_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True

        elif callb.run_squeezing and solver.frame == callb.squeezing_start_frame:

            solver.recording_path = os.path.join(output_dir,"squeezing")
            solver.record_path_has_changed = True
            check_dir_exists(solver.recording_path)
            print(f"Frame {solver.frame}: Start squeezing fames")

            V, T, F = read_mesh_file("../data/bar.mesh")
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)
            solver.reference_frame = solver.frame

            model.compute_sides_and_corner_indices()
            right_side_verts = model._side_surface_verts["right"]
            left_side_verts = model._side_surface_verts["left"]

            # Generate serise for streatching
            squeezing_motion_x_axis_right = - callb.create_xyz_stretch_motion_with_jumps(callb.number_squeezing_frames, 0,
                                                                      1,
                                                                      displacement_xyz=(0.2, 0.0, 0.0))
            squeezing_motion_x_axis_left = - squeezing_motion_x_axis_right

            for v in right_side_verts:
                model.add_positional_constraint(v, args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=squeezing_motion_x_axis_right, frame_reset=solver.frame)
                model.picked_vert[v] = True

            for v in left_side_verts:
                model.add_positional_constraint(v, args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=squeezing_motion_x_axis_left, frame_reset=solver.frame)
                model.picked_vert[v] = True

            solver.set_dirty()
            print("Squeezing - positional constraint added to right and left sides.")

        elif callb.run_squeezing and solver.frame == callb.squeezing_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True

        elif callb.run_poking and solver.frame == callb.poking_start_frame:

            solver.recording_path = os.path.join(output_dir, "poking")
            solver.record_path_has_changed = True
            check_dir_exists(solver.recording_path)
            print(f"Frame {solver.frame}: Start poking frames")

            V, T, F = read_mesh_file("../data/bar.mesh")
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)
            solver.reference_frame = solver.frame

            callb.poked_points, callb.labels = callb.compute_voronoi_seeds_incremental(model.init_positions, callb.number_poking_points, visualize=False)

            callb.poking_motion, _, _ = callb.create_poking_motions_at_given_seeds(model.positions,
                                                                        callb.poked_points,
                                                                        direction="normal",     # "normal" or "x"/"y"/"z"
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
            if record_fom_info:
                solver.store_current_snapshots = True

        elif callb.run_falling and solver.frame == callb.gravitational_fall_start_frame:

            solver.recording_path = os.path.join(output_dir,"free_falling")
            solver.record_path_has_changed = True
            check_dir_exists(solver.recording_path)
            print(f"Frame {solver.frame}: Starting free fall frames")

            V, T, F = read_mesh_file("../data/bar.mesh")
            #
            # # params.edit_system_args(args, "Bar")
            # # V, T, F, _ = get_simple_bar_model(args.bar_width, args.bar_height, args.bar_depth)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)
            solver.reference_frame = solver.frame

        elif callb.run_falling and solver.frame == callb.gravitational_fall_end_frame:
            if record_fom_info:
                solver.store_current_snapshots = True

        elif callb.run_rotating and solver.frame == callb.rotating_start_frame:

            solver.recording_path = os.path.join(output_dir, "rotating")
            solver.record_path_has_changed = True
            check_dir_exists(solver.recording_path)
            print(f"Frame {solver.frame}: Starting rotating frames")

            V, T, F = read_mesh_file("../data/bar.mesh")
            #
            # # params.edit_system_args(args, "Bar")
            # # V, T, F, _ = get_simple_bar_model(args.bar_width, args.bar_height, args.bar_depth)
            reset_simulation_model(V, F, T, should_rescale=True, hight=args.height_up_shift)
            solver.reference_frame = solver.frame

            callb.create_full_rotating_motion(V, callb.number_rotating_frames)

        elif callb.run_rotating and callb.rotating_end_frame > solver.frame > callb.rotating_start_frame:
            f = solver.frame - solver.reference_frame
            model.positions[:] = callb.rotating_positions_series[f]

        elif callb.run_rotating and solver.frame == callb.rotating_end_frame:
            print(f"Frame {solver.frame}: Ending rotating frames")

            if record_fom_info:
                solver.store_current_snapshots = True


        if solver.frame == solver.max_p_snapshots_num + 10:

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
            psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

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


def cloth_automated_callback(args, record_fom_info = False,
                                               params=None,
                                               object = "cloth",
                                               experiment="automated_deformationgradient",
                                               ):
    experiment = object + "_" + experiment
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = args.is_simulating
    output_dir = args.output_dir

    callb.set_automated_experiments(object, args)
    if callb.total_frames > args.max_p_snapshots_num:
        solver.set_max_recorded_frames(callb.total_frames)

    psim.PushItemWidth(200)
    psim.TextUnformatted("== Projective Dynamics ==")
    psim.Separator()

    def callback():
        nonlocal output_dir, is_simulating
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        # Frame 0: create mesh and apply initial constraints
        if solver.frame == 0:
            print(f"Frame {solver.frame}: Creating cloth and fixing left/right corners")

            params.edit_system_args(args, "Cloth")

            V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
            reset_simulation_model(V, F, np.empty((0, 3)), should_rescale=True)
            object_name = "cloth"

            if record_fom_info:
                output_dir = callb.make_sim_path(output_dir, solver, args, object_name, experiment, record_fom_info)
                # record parameters for tracking
                with open(output_dir + "/args.txt", "w") as f:
                    for key, value in vars(args).items():
                        f.write(f"{key}: {value}\n")

            solver.set_dirty()


def cloth_automated_bend_spring_strain_callback(args, record_fom_info = False, params=None,experiment="cloth_automated_bend_spring_strain"):
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = args.is_simulating
    output_path = args.output_dir
    def callback():
        nonlocal output_path, is_simulating
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        # Frame 0: create mesh and apply initial constraints
        if solver.frame == 0:
            print("Frame 0: Creating cloth and fixing left/right corners")

            params.edit_system_args(args, "Cloth")

            V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
            reset_simulation_model(V, F, np.empty((0, 3)), should_rescale=True)
            object_name = "cloth"

            check_dir_exists(os.path.join(output_path, object_name))
            # mesh = trimesh.Trimesh(vertices=V, faces=F)
            # mesh.export(os.path.join(output_path, object_name, object_name+".obj"))

            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            model.fix_cloth_corners(side="top")
            model.fix_cloth_corners(side="bottom")

            # Apply any desired constraints
            model.immobilize()
            model.clear_constraints()
            model.reset_constraints_attributes()

            if args.vert_bending_constraint:
                model.add_vertex_bending_constraint(args.vert_bending_constraint_wi)
            if args.edge_constraint:
                model.add_edge_spring_constrain(args.edge_constraint_wi)
            if args.tri_strain_constraint:
                model.add_tri_constrain_strain(args.sigma_min, args.sigma_max, args.strain_limit_constraint_wi)

            # if recording snapshots build output file name/ path
            if record_fom_info:
                constrproj_case = "constraint_projection/FOM"
                if solver.has_reduced_constraint_projectios:
                    constrproj_case = "constraint_projection/" + args.constraint_projection_basis_type

                specify_path = ""
                if model.has_verts_bending_constraints:
                    specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
                    if args.vert_bending_reduced:
                        specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"

                if model.has_edge_spring_constraints:
                    specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
                    if args.edge_spring_reduced:
                        specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"

                if model.has_tris_strain_constraints:
                    specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tri_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
                if model.has_tets_strain_constraints:
                    specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tet_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
                if model.has_tets_deformation_gradient_constraints:
                    specify_path = specify_path + "tets_deformation_gradient_wi" + str(
                        args.deformation_gradient_constraint_wi) + "_"
                    if args.tet_deformation_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components)+ "_"

                output_path += "/" + object_name + "/" + experiment + "/" + "/" + constrproj_case + "/" + specify_path + "/"
                check_dir_exists(output_path)

                solver.set_record_path(output_path)
                solver.set_store_p(record_fom_info)
            solver.set_dirty()

        elif solver.frame == 20:
            print("Frame 10: Releasing left corners")
            model.release_cloth_corners(side="bottom")

        elif solver.frame == 60:
            print("Frame 20: Fixing left, releasing right")
            model.fix_cloth_corners(side="bottom")
            model.release_cloth_corners(side="top")

        elif solver.frame == 140:
            print("Frame 30: Releasing all corners")
            model.release_cloth_corners(side="top")
            model.release_cloth_corners(side="bottom")
            model.fix_cloth_corners(side="right")


        elif solver.frame == 240:
            print("Stopping simulation.")
            is_simulating = False
            ps.unshow()
            return

        # Run a single simulation step
        if model is not None and is_simulating:

            pre_draw_handler = PreDrawHandler(
                lambda: model.positions.shape[0] > 0, args, solver, fext,
                record_info=record_fom_info, record_path=output_path
            )
            pre_draw_handler.set_animating(True)
            pre_draw_handler.handle()

        if model is not None:
            psim.BulletText(f"Vertices: {model.positions.shape[0]}")
            psim.BulletText(f"Triangles: {model.faces.shape[0]}")
            psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
            psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

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

def cloth_automated_strain_callback(args, record_fom_info = False, params=None,experiment="cloth_automated_strain"):
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = args.is_simulating
    output_path = args.output_dir
    def callback():
        nonlocal output_path, is_simulating
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        # Frame 0: create mesh and apply initial constraints
        if solver.frame == 0:
            print("Frame 0: Creating cloth and fixing left/right corners")

            params.edit_system_args(args, "Cloth")

            V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
            reset_simulation_model(V, F, np.empty((0, 3)), should_rescale=True)
            object_name = "cloth"

            check_dir_exists(os.path.join(output_path, object_name))
            # mesh = trimesh.Trimesh(vertices=V, faces=F)
            # mesh.export(os.path.join(output_path, object_name, object_name+".obj"))

            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            # model.fix_surface_side_vertices(side="right")
            # model.fix_surface_side_vertices(side="left")

            # Apply any desired constraints
            model.immobilize()
            model.clear_constraints()
            model.reset_constraints_attributes()

            if args.vert_bending_constraint:
                model.add_vertex_bending_constraint(args.vert_bending_constraint_wi)
            if args.edge_constraint:
                model.add_edge_spring_constrain(args.edge_constraint_wi)
            if args.tri_strain_constraint:
                model.add_tri_constrain_strain(args.sigma_min, args.sigma_max, args.strain_limit_constraint_wi)

            # if recording snapshots build output file name/ path
            if record_fom_info:
                constrproj_case = "constraint_projection/FOM"
                if solver.has_reduced_constraint_projectios:
                    constrproj_case = "constraint_projection/" + args.constraint_projection_basis_type

                specify_path = ""
                if model.has_verts_bending_constraints:
                    specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
                    if args.vert_bending_reduced:
                        specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"

                if model.has_edge_spring_constraints:
                    specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
                    if args.edge_spring_reduced:
                        specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"

                if model.has_tris_strain_constraints:
                    specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tri_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
                if model.has_tets_strain_constraints:
                    specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tet_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
                if model.has_tets_deformation_gradient_constraints:
                    specify_path = specify_path + "tets_deformation_gradient_wi" + str(
                        args.deformation_gradient_constraint_wi) + "_"
                    if args.tet_deformation_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components)+ "_"

                output_path += "/" + object_name + "/" + experiment + "/" + "/" + constrproj_case + "/" + specify_path + "/"
                check_dir_exists(output_path)

                solver.set_record_path(output_path)
                solver.set_store_p(record_fom_info)
            solver.set_dirty()

        # elif solver.frame == 20:
        #     print("Frame 10: Releasing sides")
        #     model.release_surface_side_vertices(side="right")
        #     model.release_surface_side_vertices(side="left")


        elif solver.frame == 220:
            print("Stopping simulation.")
            is_simulating = False
            ps.unshow()
            return

        # Run a single simulation step
        if model is not None and is_simulating:

            pre_draw_handler = PreDrawHandler(
                lambda: model.positions.shape[0] > 0, args, solver, fext,
                record_info=record_fom_info, record_path=output_path
            )
            pre_draw_handler.set_animating(True)
            pre_draw_handler.handle()

        if model is not None:
            psim.BulletText(f"Vertices: {model.positions.shape[0]}")
            psim.BulletText(f"Triangles: {model.faces.shape[0]}")
            psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
            psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

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

def cloth_automated_bend_callback(args, record_fom_info = False, params=None,experiment="cloth_automated_bend"):
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = args.is_simulating
    output_path = args.output_dir
    def callback():
        nonlocal output_path, is_simulating
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        # Frame 0: create mesh and apply initial constraints
        if solver.frame == 0:
            print("Frame 0: Creating cloth and fixing left/right corners")

            params.edit_system_args(args, "Cloth")

            V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
            reset_simulation_model(V, F, np.empty((0, 3)), should_rescale=True)
            object_name = "cloth"

            check_dir_exists(os.path.join(output_path, object_name))
            # mesh = trimesh.Trimesh(vertices=V, faces=F)
            # mesh.export(os.path.join(output_path, object_name, object_name+".obj"))

            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            # model.fix_surface_side_vertices(side="right")
            # model.fix_surface_side_vertices(side="left")

            # Apply any desired constraints
            model.immobilize()
            model.clear_constraints()
            model.reset_constraints_attributes()

            if args.vert_bending_constraint:
                model.add_vertex_bending_constraint(args.vert_bending_constraint_wi)
            if args.edge_constraint:
                model.add_edge_spring_constrain(args.edge_constraint_wi)
            if args.tri_strain_constraint:
                model.add_tri_constrain_strain(args.sigma_min, args.sigma_max, args.strain_limit_constraint_wi)

            # if recording snapshots build output file name/ path
            if record_fom_info:
                constrproj_case = "constraint_projection/FOM"
                if solver.has_reduced_constraint_projectios:
                    constrproj_case = "constraint_projection/" + args.constraint_projection_basis_type

                specify_path = ""
                if model.has_verts_bending_constraints:
                    specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
                    if args.vert_bending_reduced:
                        specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"

                if model.has_edge_spring_constraints:
                    specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
                    if args.edge_spring_reduced:
                        specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"

                if model.has_tris_strain_constraints:
                    specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tri_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
                if model.has_tets_strain_constraints:
                    specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tet_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
                if model.has_tets_deformation_gradient_constraints:
                    specify_path = specify_path + "tets_deformation_gradient_wi" + str(
                        args.deformation_gradient_constraint_wi) + "_"
                    if args.tet_deformation_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components)+ "_"

                output_path += "/" + object_name + "/" + experiment + "/" + "/" + constrproj_case + "/" + specify_path + "/"
                check_dir_exists(output_path)

                solver.set_record_path(output_path)
                solver.set_store_p(record_fom_info)
            solver.set_dirty()

        # elif solver.frame == 20:
        #     print("Frame 10: Releasing sides")
        #     model.release_surface_side_vertices(side="right")
        #     model.release_surface_side_vertices(side="left")


        elif solver.frame == 55:
            print("Stopping simulation.")
            is_simulating = False
            ps.unshow()
            return

        # Run a single simulation step
        if model is not None and is_simulating:

            pre_draw_handler = PreDrawHandler(
                lambda: model.positions.shape[0] > 0, args, solver, fext,
                record_info=record_fom_info, record_path=output_path
            )
            pre_draw_handler.set_animating(True)
            pre_draw_handler.handle()

        if model is not None:
            psim.BulletText(f"Vertices: {model.positions.shape[0]}")
            psim.BulletText(f"Triangles: {model.faces.shape[0]}")
            psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
            psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

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

def cloth_test(args, record_fom_info = False, params=None,experiment="cloth_automated_bend_spring_strain_test"):
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = True
    output_path = args.output_dir

    start_frame = 0
    bottom_side_verts = None
    counter = 0
    final_frame = 100


    def create_rotation_around_arbitrary_axis(
            axis_vector,
            axis_point=np.zeros(3),
            start_frame=0,
            num_frames=30,
            num_rotations=1,
            total_frames=None
    ):
        """
        Rotate a point around an arbitrary axis in space over time.

        Parameters:
        - point: np.array(3,), the vertex to rotate
        - axis_vector: np.array(3,), direction vector of the axis
        - axis_point: np.array(3,), a point the axis passes through
        - start_frame: int, first frame to apply rotation
        - num_frames: int, number of frames over which rotation occurs
        - num_rotations: int, how many full 360° rotations
        - total_frames: int, total number of frames in animation

        Returns:
        - motion: (total_frames, 3) array of vertex positions over time
        """

        if total_frames is None:
            total_frames = start_frame + num_frames

        motion = np.zeros((total_frames, 3))

        # Normalize the axis direction
        axis_dir = axis_vector / np.linalg.norm(axis_vector)

        # Generate rotation angles
        angles = np.linspace(0, 2 * np.pi * num_rotations, num_frames, endpoint=False)

        for i, theta in enumerate(angles):
            # Rodrigues' rotation formula
            v = -axis_point  # vector from axis_point to point
            k = axis_dir

            v_rot = (
                    v * np.cos(theta)
                    + np.cross(k, v) * np.sin(theta)
                    + k * np.dot(k, v) * (1 - np.cos(theta))
            )
            rotated = axis_point + v_rot
            motion[start_frame + i] = rotated

        return motion

    def callback():
        nonlocal output_path, is_simulating, bottom_side_verts, counter
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        counter += 1
        # Frame 0: create mesh and apply initial constraints
        if solver.frame == 0:
            print(f"Frame: {solver.frame} Creating cloth and generating poking fames")

            params.edit_system_args(args, "Cloth")

            V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
            reset_simulation_model(V, F, np.empty((0, 3)), should_rescale=True)
            object_name = "cloth"

            check_dir_exists(os.path.join(output_path, object_name))
            # mesh = trimesh.Trimesh(vertices=V, faces=F)
            # mesh.export(os.path.join(output_path, object_name, object_name + ".obj"))

            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            top_side_verts = model.fix_surface_side_vertices(side="top", fix_it = True, return_target = True)  # fix
            # return indices and not fix
            bottom_side_verts = model.fix_surface_side_vertices(side="bottom", fix_it = False, return_target = True)


            # Apply any desired constraints
            model.immobilize()
            model.clear_constraints()
            model.reset_constraints_attributes()

            print("Poking - positional constraint added to center vertex")
            for i in range(len(bottom_side_verts)):
                center_i = top_side_verts[i]
                initial_i = bottom_side_verts[i]  # point on x-axis
                rotation_series = create_rotation_around_arbitrary_axis(
                    axis_vector=np.array([1.5, 0, 0]),
                    axis_point=V[center_i],
                    start_frame=10,
                    num_frames=60,
                    num_rotations=1,
                    total_frames=100)
                model.add_positional_constraint(initial_i, args.positional_constraint_wi,
                                                motion_type="user_defined", frames_series=rotation_series)

                model.picked_vert[initial_i] = True

            if args.vert_bending_constraint:
                model.add_vertex_bending_constraint(args.vert_bending_constraint_wi)
            if args.edge_constraint:
                model.add_edge_spring_constrain(args.edge_constraint_wi)
            if args.tri_strain_constraint:
                model.add_tri_constrain_strain(args.sigma_min, args.sigma_max, args.strain_limit_constraint_wi)

            # if recording snapshots build output file name/ path
            if record_fom_info:
                constrproj_case = "constraint_projection/FOM"
                if solver.has_reduced_constraint_projectios:
                    constrproj_case = "constraint_projection/" + args.constraint_projection_basis_type

                specify_path = ""
                if model.has_verts_bending_constraints:
                    specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
                    if args.vert_bending_reduced:
                        specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"

                if model.has_edge_spring_constraints:
                    specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
                    if args.edge_spring_reduced:
                        specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"

                if model.has_tris_strain_constraints:
                    specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tri_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
                if model.has_tets_strain_constraints:
                    specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tet_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
                if model.has_tets_deformation_gradient_constraints:
                    specify_path = specify_path + "tets_deformation_gradient_wi" + str(
                        args.deformation_gradient_constraint_wi) + "_"
                    if args.tet_deformation_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components) + "_"

                output_path += "/" + object_name + "/" + experiment + "/" + "/" + constrproj_case + "/" + specify_path + "/"
                check_dir_exists(output_path)

                solver.set_record_path(output_path)
                solver.set_store_p(record_fom_info)
            solver.set_dirty()

        if counter == final_frame:
            print("Stopping simulation.")
            is_simulating = False
            ps.unshow()
            return

        # Run a single simulation step
        if model is not None and is_simulating:
            pre_draw_handler = PreDrawHandler(
                lambda: model.positions.shape[0] > 0, args, solver, fext,
                record_info=record_fom_info, record_path=output_path
            )
            pre_draw_handler.set_animating(True)
            pre_draw_handler.handle()

        if model is not None:
            psim.BulletText(f"Vertices: {model.positions.shape[0]}")
            psim.BulletText(f"Triangles: {model.faces.shape[0]}")
            psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
            psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

            if model.has_verts_bending_constraints:
                psim.BulletText(f"Vertices bending constraint: {len(model.verts_bending_constraints)}")
                psim.BulletText(f"wi: {str(args.vert_bending_constraint_wi)}")

            if model.has_edge_spring_constraints:
                psim.BulletText(f"Edge pring constraint: {len(model.edge_spring_constraints)}")
                psim.BulletText(f"wi: {str(args.edge_constraint_wi)}")

            if model.has_tris_strain_constraints:
                psim.BulletText(f"Triangles strain constraint: {len(model.tris_strain_constraints)}")
                psim.BulletText(f"wi: {str(args.strain_limit_constraint_wi)}")

        psim.End()

    return callback


def cloth_snapshots(args, record_fom_info = False, params=None,experiment="cloth_automated_bend_spring_strain_snapshots"):
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = True
    output_path = args.output_dir

    start_poking_frame = 0
    poking_half_width = 0.2  # how far we poke from the original point *[-1, 1]
    poking_frames_per_point = 20
    rest_frames_per_point = 10
    poking_series = None
    poked_points = None
    top_side_verts = None
    bottom_side_verts = None
    right_side_verts = None
    left_side_verts = None
    number_pockes = 10
    total_frames_poking_frames = number_pockes *(rest_frames_per_point + poking_frames_per_point)
    free_fall_frames = 5
    start_stretching_frame = total_frames_poking_frames + free_fall_frames
    stretching_frames = 20
    number_stretches = 2
    rest_frames_per_stretch = 10
    release_stretching_frame = start_stretching_frame + 2* stretching_frames
    start_bottom_top_rolling_frame = release_stretching_frame + rest_frames_per_stretch
    rolling_frames = 100


    counter = 0
    final_frame = 600 #total_frames_poking_frames +free_fall_frames + stretching_frames
    number_recorded_frames = final_frame - 4

    def create_poke_z_motion_with_jumps(f_l, f_j, k, z_range=1.0):
        """
        Generate a z-motion that repeats k times:
        motion phase: 0 → -z → +z → -z over f_l frames
        pause phase: zeros for f_j frames

        :param f_l: Frames per motion cycle
        :param f_j: Frames per jump (pause)
        :param k: Number of motion + pause cycles
        :param z_range: Peak z-displacement
        :return: (total_frames, 3) array of z-motion per frame (x, y = 0)
        """
        motion_pattern = []
        for _ in range(k):
            # Motion part: 0 → -z → +z → -z over f_l frames
            quarter = f_l // 4
            z = z_range
            z_values = np.concatenate([
                np.linspace(0, -z, quarter, endpoint=False),
                np.linspace(-z, +z, quarter, endpoint=False),
                np.linspace(+z, -z, quarter, endpoint=False),
                np.linspace(-z, 0, f_l - 3 * quarter)  # ensure total = f_l
            ])

            # Pause part: f_j frames of zero
            pause_values = np.zeros(f_j)

            motion_pattern.append(z_values)
            motion_pattern.append(pause_values)

        z_all = np.concatenate(motion_pattern)

        # Make (f_total, 3) motion array (x, y = 0)
        motion = np.zeros((len(z_all), 3))
        motion[:, 2] = z_all

        return motion
    def get_voronoi_seeds_and_partition(V, F, k, visualize=True):
        """
        Return the center vertex and k seeds for Voronoi partitioning based on geodesic-like distances.

        Parameters:
            V (n,3): Vertex positions
            F (m,3): Triangle indices
            k (int): Number of Voronoi seeds
            visualize (bool): Whether to visualize the partitioning in 2D

        Returns:
            seeds (k+1,): List of vertex indices: [center_idx, seed_1, ..., seed_k]
            labels (n,): Voronoi region label for each vertex
        """

        # Compute global center (Euclidean)
        center_2d = V[:, :2].mean(axis=0)
        dists = np.linalg.norm(V[:, :2] - center_2d, axis=1)
        center_idx = np.argmin(dists)

        # Sample seeds using the furthest point sampling
        seeds = [center_idx]
        remaining = set(range(V.shape[0]))
        remaining.remove(center_idx)

        for _ in range(k):
            dist_to_seeds = np.min(distance_matrix(V[:, :2], V[seeds, :2]), axis=1)
            dist_to_seeds[seeds] = -1  # mask already chosen
            new_seed = np.argmax(dist_to_seeds)
            seeds.append(new_seed)

        seeds = np.array(seeds)

        # Assign labels based on nearest seed (Euclidean for simplicity)
        dist_to_seeds = distance_matrix(V[:, :2], V[seeds, :2])
        labels = np.argmin(dist_to_seeds, axis=1)

        # Visualization (2D projection)
        if visualize:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(V[:, 0], V[:, 1], c=labels, s=10, cmap="tab20")
            plt.scatter(V[seeds, 0], V[seeds, 1], c='black', s=50, marker='x', label="Voronoi Seeds")
            plt.title("Voronoi Partitioning (Euclidean Approximation)")
            plt.axis("equal")
            plt.legend()
            plt.colorbar(scatter, label="Voronoi Region")
            plt.show()

        return seeds, labels

    def compute_voronoi_seeds_incremental(positions, k, start_idx=None, visualize=True, title="Voronoi Partitioning (Euclidean Approximation)"):
        """
        Select k Voronoi seeds on a mesh using incremental farthest-point sampling.

        Parameters:
            positions: (n, 3) numpy array of vertex positions
            k: int, number of seeds to return
            start_idx: optional int, index of first seed (default = closest to centroid)

        Returns:
            seeds: list of k vertex indices
        """
        n = positions.shape[0]

        # Start from the closest point to the centroid if not given
        if start_idx is None:
            center = positions.mean(axis=0)
            start_idx = np.argmin(np.linalg.norm(positions - center, axis=1))

        seeds = [start_idx]
        dists = np.linalg.norm(positions - positions[start_idx], axis=1)

        for _ in range(1, k):
            # Find the point with maximum distance to the nearest seed
            min_dists = dists
            next_idx = np.argmax(min_dists)
            seeds.append(next_idx)

            # Update minimum distances to the new seed
            new_dists = np.linalg.norm(positions - positions[next_idx], axis=1)
            dists = np.minimum(dists, new_dists)

        # Assign each vertex to the nearest seed
        tree = cKDTree(positions[seeds])
        labels = tree.query(positions)[1]  # nearest seed index
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=labels, cmap='tab20', s=5)
            ax.scatter(positions[seeds, 0], positions[seeds, 1], positions[seeds, 2], c='black', s=30, label="Seeds")
            ax.set_title(title)
            ax.legend()
            plt.show()

        return np.array(seeds), labels

    def create_xyz_stretch_motion_with_jumps(f_l, f_j, k, displacement_xyz=(1.0, 0.0, 0.0)):
        """
        Generate a multi-axis motion that repeats k times:
          - motion phase: linearly interpolate from 0 to target displacement and back over f_l frames
          - pause phase: hold at rest (zeros) for f_j frames

        :param f_l: Frames per motion cycle (excluding jump)
        :param f_j: Frames per pause (jump)
        :param k: Number of motion+pause cycles
        :param displacement_xyz: (dx, dy, dz) tuple of peak displacement along each axis
        :return: (total_frames, 3) array of displacement per frame
        """
        dx, dy, dz = displacement_xyz
        motion = []

        for _ in range(k):
            # -- Motion phase: 0 -> displacement -> 0
            half = f_l // 2
            phase1 = np.linspace(0, 1, half, endpoint=False)
            phase2 = np.linspace(1, 0, f_l - half)

            motion_phase = np.concatenate([phase1, phase2])[:, None]  # shape (f_l, 1)
            disp_phase = motion_phase * np.array([[dx, dy, dz]])  # broadcast to (f_l, 3)

            # -- Pause phase: hold at zero
            pause_phase = np.zeros((f_j, 3))

            # -- Append both
            motion.append(disp_phase)
            motion.append(pause_phase)

        motion_array = np.concatenate(motion, axis=0)  # shape: (k * (f_l + f_j), 3)
        return motion_array

    def create_rotation_around_arbitrary_axis(
            axis_vector,
            axis_point=np.zeros(3),
            start_frame=0,
            num_frames=30,
            num_rotations=1,
            total_frames=None
    ):
        """
        Rotate a point around an arbitrary axis in space over time.

        Parameters:
        - point: np.array(3,), the vertex to rotate
        - axis_vector: np.array(3,), direction vector of the axis
        - axis_point: np.array(3,), a point the axis passes through
        - start_frame: int, first frame to apply rotation
        - num_frames: int, number of frames over which rotation occurs
        - num_rotations: int, how many full 360° rotations
        - total_frames: int, total number of frames in animation

        Returns:
        - motion: (total_frames, 3) array of vertex positions over time
        """

        if total_frames is None:
            total_frames = start_frame + num_frames

        motion = np.zeros((total_frames, 3))

        # Normalize the axis direction
        axis_dir = axis_vector / np.linalg.norm(axis_vector)

        # Generate rotation angles
        angles = np.linspace(0, 2 * np.pi * num_rotations, num_frames, endpoint=False)

        for i, theta in enumerate(angles):
            # Rodrigues' rotation formula
            v = -axis_point  # vector from axis_point to point
            k = axis_dir

            v_rot = (
                    v * np.cos(theta)
                    + np.cross(k, v) * np.sin(theta)
                    + k * np.dot(k, v) * (1 - np.cos(theta))
            )
            rotated = axis_point + v_rot
            motion[start_frame + i] = rotated

        return motion

    def callback():
        nonlocal output_path, is_simulating, poking_series, poked_points, poking_frames_per_point, rest_frames_per_point, number_pockes, counter,\
            top_side_verts, bottom_side_verts, right_side_verts, left_side_verts
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        counter += 1
        # Frame 0: create mesh and apply initial constraints
        if solver.frame == 0:
            print(f"Frame: {solver.frame} Creating cloth and generating poking fames")

            params.edit_system_args(args, "Cloth")

            V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
            reset_simulation_model(V, F, None, should_rescale=True)
            object_name = "cloth"

            check_dir_exists(os.path.join(output_path, object_name))
            # mesh = trimesh.Trimesh(vertices=V, faces=F)
            # mesh.export(os.path.join(output_path, object_name, object_name + ".obj"))

            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            model.fix_surface_side_vertices(args.positional_constraint_wi, side="top")
            # model.fix_cloth_corners(side="bottom")

            # find the closest vertex to center

            # Generate motion serise for poking
            # Generate z values: 0 → 1 ->> -1 linearly
            # poked_points, labels = get_voronoi_seeds_and_partition(V, F, number_pockes)
            poked_points, lables = compute_voronoi_seeds_incremental(V, number_pockes, visualize=False)
            poking_series = create_poke_z_motion_with_jumps(poking_frames_per_point, rest_frames_per_point, poked_points.shape[0], z_range=poking_half_width)

            # How many frames to record
            solver.set_max_recorded_frames(number_recorded_frames)

            # Apply any desired constraints
            model.immobilize()
            model.clear_constraints()
            model.reset_constraints_attributes()

            model.add_positional_constraint(poked_points[0], args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=poking_series)
            print("Poking - positional constraint added to center vertex")

            model.picked_vert[poked_points[0]] = True
            if args.vert_bending_constraint:
                model.add_vertex_bending_constraint(args.vert_bending_constraint_wi)
            if args.edge_constraint:
                model.add_edge_spring_constrain(args.edge_constraint_wi)
            if args.tri_strain_constraint:
                model.add_tri_constrain_strain(args.sigma_min, args.sigma_max, args.strain_limit_constraint_wi)

            # if recording snapshots build output file name/ path
            if record_fom_info:
                constrproj_case = "constraint_projection/FOM"
                if solver.has_reduced_constraint_projections:
                    constrproj_case = "constraint_projection/" + args.constraint_projection_basis_type

                specify_path = ""
                if model.has_verts_bending_constraints:
                    specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
                    if args.vert_bending_reduced:
                        specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"

                if model.has_edge_spring_constraints:
                    specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
                    if args.edge_spring_reduced:
                        specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"

                if model.has_tris_strain_constraints:
                    specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tri_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
                if model.has_tets_strain_constraints:
                    specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tet_strain_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
                if model.has_tets_deformation_gradient_constraints:
                    specify_path = specify_path + "tets_deformation_gradient_wi" + str(
                        args.deformation_gradient_constraint_wi) + "_"
                    if args.tet_deformation_reduced:
                        specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components) + "_"

                output_path += "/" + object_name + "/" + experiment + "/" + "/" + constrproj_case + "/" + specify_path + "/"
                check_dir_exists(output_path)

                solver.set_record_path(output_path)
                solver.set_store_p(record_fom_info)
            solver.set_dirty()

        elif solver.frame % (poking_frames_per_point + rest_frames_per_point) == 1:
            i = solver.frame // (poking_frames_per_point + rest_frames_per_point)

            if i < poked_points.shape[0]:
                model.add_positional_constraint(poked_points[i], args.positional_constraint_wi,
                                                motion_type="user_defined", frames_series=poking_series)
                model.picked_vert[poked_points[i]] = True
                solver.set_dirty()
                print(f"Poking - positional constraint added to {i} vertex")

        elif solver.frame % (poking_frames_per_point + rest_frames_per_point) == poking_frames_per_point\
                and 0 < solver.frame < total_frames_poking_frames:
            i = solver.frame // (poking_frames_per_point + rest_frames_per_point)
            if i < poked_points.shape[0]:
                print(f"Removing - positional constraint remover from {i} vertex")
                model.remove_positional_constraint(poked_points[i])
                model.picked_vert[poked_points[i]] = False
                solver.set_dirty()

        elif solver.frame == total_frames_poking_frames:
            model.release_surface_side_vertices(side="top")

        elif solver.frame == start_stretching_frame:
            print(f"Frame: {solver.frame} Resetting cloth mesh and generating top-bottom stretching fames")

            V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
            reset_simulation_model(V, F, np.empty((0, 3)), should_rescale=True)
            object_name = "cloth"

            check_dir_exists(os.path.join(output_path, object_name))
            # mesh = trimesh.Trimesh(vertices=V, faces=F)

            # Generate serise for streatching
            stretch_motion_top = create_xyz_stretch_motion_with_jumps(stretching_frames, rest_frames_per_stretch,
                                                                      number_stretches, displacement_xyz=(0.0, 0.4, 0.0))
            stretch_motion_bottom = - stretch_motion_top

            # Apply any desired constraints
            model.immobilize()
            model.clear_constraints()
            model.reset_constraints_attributes()
            #
            model.compute_sides_and_corner_indices()
            top_side_verts = model._side_surface_verts["top"]
            bottom_side_verts = model._side_surface_verts["bottom"]
            # top_side_verts = model.fix_surface_side_vertices(side="top", return_target=True)
            # bottom_side_verts = model.fix_surface_side_vertices(side="bottom", return_target=True)

            for v in top_side_verts:
                model.add_positional_constraint(v, args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=stretch_motion_top, frame_reset=solver.frame)
                model.picked_vert[v] = True

            for v in bottom_side_verts:
                model.add_positional_constraint(v, args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=stretch_motion_bottom, frame_reset=solver.frame)
                model.picked_vert[v] = True

            print("Stretching - positional constraint added to top and bottom sides.")

            if args.vert_bending_constraint:
                model.add_vertex_bending_constraint(args.vert_bending_constraint_wi)
            if args.edge_constraint:
                model.add_edge_spring_constrain(args.edge_constraint_wi)
            if args.tri_strain_constraint:
                model.add_tri_constrain_strain(args.sigma_min, args.sigma_max, args.strain_limit_constraint_wi)

            solver.set_dirty()

        elif solver.frame == start_stretching_frame + stretching_frames:

            # model.release_surface_side_vertices(side="right")
            # model.release_surface_side_vertices(side="left")
            print(f"Removing - positional constraint remover from top-bottom sides.")
            for v in top_side_verts:
                model.remove_positional_constraint(v)
                model.picked_vert[v] = False

            for v in bottom_side_verts:
                model.remove_positional_constraint(v)
                model.picked_vert[v] = False

            print(f"Frame: {solver.frame} Resetting cloth mesh and generating left-right stretching fames")

            right_side_verts = model._side_surface_verts["right"]
            left_side_verts = model._side_surface_verts["left"]
            # right_side_verts = model.fix_surface_side_vertices(side="right", return_target=True)
            # left_side_verts = model.fix_surface_side_vertices(side="left", return_target=True)

            # Generate serise for streatching
            stretch_motion_right = create_xyz_stretch_motion_with_jumps(stretching_frames, rest_frames_per_stretch,
                                                                      number_stretches,
                                                                      displacement_xyz=(0.4, 0.0, 0.0))
            stretch_motion_left = - stretch_motion_right
            for v in right_side_verts:
                model.add_positional_constraint(v, args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=stretch_motion_right, frame_reset=solver.frame)
                model.picked_vert[v] = True

            for v in left_side_verts:
                model.add_positional_constraint(v, args.positional_constraint_wi,
                                            motion_type="user_defined", frames_series=stretch_motion_left, frame_reset=solver.frame)
                model.picked_vert[v] = True

            solver.set_dirty()
            print("Stretching - positional constraint added to right and left sides.")

        elif solver.frame == release_stretching_frame:

            print(f"Removing - positional constraint remover from top-bottom sides.")
            for v in right_side_verts:
                model.remove_positional_constraint(v)
                model.picked_vert[v] = False

            for v in left_side_verts:
                model.remove_positional_constraint(v)
                model.picked_vert[v] = False
            solver.set_dirty()

        elif solver.frame == start_bottom_top_rolling_frame:
            print(f"Frame: {solver.frame} Creating cloth and generating poking fames")

            params.edit_system_args(args, "Cloth")

            V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
            reset_simulation_model(V, F, np.empty((0, 3)), should_rescale=True)
            object_name = "cloth"

            check_dir_exists(os.path.join(output_path, object_name))
            # mesh = trimesh.Trimesh(vertices=V, faces=F)
            # mesh.export(os.path.join(output_path, object_name, object_name + ".obj"))

            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            top_side_verts = model.fix_surface_side_vertices(args.positional_constraint_wi, side="top", fix_it=True, return_target=True)  # fix
            # return indices and not fix
            bottom_side_verts = model.fix_surface_side_vertices(args.positional_constraint_wi, side="bottom", fix_it=False, return_target=True)

            # Apply any desired constraints
            model.immobilize()
            model.clear_constraints()
            model.reset_constraints_attributes()

            print("Poking - positional constraint added to center vertex")
            for i in range(len(bottom_side_verts)):
                center_i = top_side_verts[i]
                initial_i = bottom_side_verts[i]  # point on x-axis
                rotation_series = create_rotation_around_arbitrary_axis(
                    axis_vector=np.array([1.5, 0, 0]),
                    axis_point=V[center_i],
                    start_frame=10,
                    num_frames=60,
                    num_rotations=1,
                    total_frames=100)
                model.add_positional_constraint(initial_i, args.positional_constraint_wi,
                                                motion_type="user_defined", frames_series=rotation_series, frame_reset=solver.frame)

                model.picked_vert[initial_i] = True

            if args.vert_bending_constraint:
                model.add_vertex_bending_constraint(args.vert_bending_constraint_wi)
            if args.edge_constraint:
                model.add_edge_spring_constrain(args.edge_constraint_wi)
            if args.tri_strain_constraint:
                model.add_tri_constrain_strain(args.sigma_min, args.sigma_max, args.strain_limit_constraint_wi)

            solver.set_dirty()

        if counter == final_frame:
            print("Stopping simulation.")
            is_simulating = False
            ps.unshow()
            return

        # Run a single simulation step
        if model is not None and is_simulating:

            pre_draw_handler = PreDrawHandler(
                lambda: model.positions.shape[0] > 0, args, solver, fext,
                record_info=record_fom_info, record_path=output_path
            )
            pre_draw_handler.set_animating(True)
            pre_draw_handler.handle()

        if model is not None:
            psim.BulletText(f"Vertices: {model.positions.shape[0]}")
            psim.BulletText(f"Triangles: {model.faces.shape[0]}")
            psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
            if model.elements is not None:
                psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

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

def interacrive_testing_callback(args, record_fom_info = False, params=None, experiment="testing"):
    global model, fext, solver, mouse_down_handler, mouse_move_handler
    solver = get_solver_class_from_name(args)
    is_simulating = args.is_simulating
    output_path = args.output_dir

    def callback():
        nonlocal output_path
        psim.PushItemWidth(200)
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()
        object_name = ""

        record = False
        system_name = "User_defined"

        def make_sim_path(args):
            nonlocal output_path

            sim_case = "FOM"

            if solver.has_reduced_position and not solver.has_reduced_constraint_projections:
                sim_case = "positions_reduced/" + args.position_basis_type
            elif solver.has_reduced_constraint_projections and not solver.has_reduced_position:
                sim_case = "constraint_projections_reduced/" + args.constraint_projection_basis_type

            elif solver.has_reduced_constraint_projections and solver.has_reduced_position:
                sim_case = "positions_and_constraint_projections_reduced/" + args.position_basis_type + "_" + args.constraint_projection_basis_type

            specify_path = ""
            if model.has_verts_bending_constraints:
                specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
                if args.vert_bending_reduced:
                    specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"

            if model.has_edge_spring_constraints:
                specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
                if args.edge_spring_reduced:
                    specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"

            if model.has_tris_strain_constraints:
                specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                if args.tri_strain_reduced:
                    specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
            if model.has_tets_strain_constraints:
                specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                if args.tet_strain_reduced:
                    specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
            if model.has_tets_deformation_gradient_constraints:
                specify_path = specify_path + "tets_deformation_gradient_wi" + str(
                    args.deformation_gradient_constraint_wi) + "_"
                if args.tet_deformation_reduced:
                    specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components) + "_"
            output_path += "/" + object_name + "/" + experiment + "/" + sim_case + "/" + specify_path + "/"
            check_dir_exists(output_path)

            solver.set_record_path(output_path)
            solver.set_store_p(record_fom_info)
            solver.set_store_q(record_fom_info)

        if solver.frame == 0:
            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            if system_name == "Bar":

                V, T, F = read_mesh_file("../data/bar.mesh")
                reset_simulation_model(V, F, T, should_rescale=True)
                object_name = "bar"

                if record_fom_info:
                    make_sim_path(args)
                    # record parameters for tracking
                    with open(output_path + "/args.txt", "w") as f:
                        for key, value in vars(args).items():
                            f.write(f"{key}: {value}\n")

            if system_name == "Cloth":

                V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
                reset_simulation_model(V, F, None, should_rescale=True)
                object_name = "Cloth"

                if record_fom_info:
                    make_sim_path(args)
                    # record parameters for tracking
                    with open(output_path + "/args.txt", "w") as f:
                        for key, value in vars(args).items():
                            f.write(f"{key}: {value}\n")

            if system_name == "User_defined":
                V, T, F = read_mesh_file("../data/sphere.mesh")
                reset_simulation_model(V, F, T, should_rescale=True)
                object_name = "sphere"

                if record_fom_info:
                    make_sim_path(args)
                    # record parameters for tracking
                    with open(output_path + "/args.txt", "w") as f:
                        for key, value in vars(args).items():
                            f.write(f"{key}: {value}\n")
                model.fix_surface_side_vertices(args.positional_constraint_wi, side="left")

            solver.set_dirty()

        if model is not None:
            set_up_mouse_handler(args, model, fext)
            psim.BulletText(f"Vertices: {model.positions.shape[0]}")
            psim.BulletText(f"Triangles: {model.faces.shape[0]}")
            psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
            psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

        if model is not None and is_simulating:

            pre_draw_handler = PreDrawHandler(
                lambda: model.positions.shape[0] > 0, args, solver, fext,
                record_info=record_fom_info, record_path=output_path
            )
            pre_draw_handler.set_animating(True)
            pre_draw_handler.handle()
        # if psim.CollapsingHeader("Physics"):
        #     if psim.TreeNode("Constraints"):
        #
        #         if object_name == "Bar":
        #             changed, args.fix_left_side = psim.Checkbox("Fix Left\nVertices Side", args.fix_left_side)
        #             changed, args.fix_right_side = psim.Checkbox("Fix Right\nVertices Side", args.fix_right_side)
        #
        #         if object_name == "Cloth":
        #             changed, args.fix_left_corners = psim.Checkbox("Fix Left\nCorners Side", args.fix_left_corners)
        #             changed, args.fix_right_corners = psim.Checkbox("Fix Right\nCorners Side", args.fix_right_corners)
        #
        #             changed, args.fix_top_corners = psim.Checkbox("Fix Top\nCorners Side", args.fix_top_corners)
        #             changed, args.fix_bottom_corners = psim.Checkbox("Fix Bottom\nCorners Side",
        #                                                              args.fix_bottom_corners)
        #
        #         changed, args.vert_bending_constraint_wi = psim.InputFloat("wi \nVertBend",
        #                                                                    args.vert_bending_constraint_wi)
        #         changed, args.vert_bending_constraint = psim.Checkbox("Active \nVertBend", args.vert_bending_constraint)
        #
        #         changed, args.edge_constraint_wi = psim.InputFloat("wi \nEdgeSpring", args.edge_constraint_wi)
        #         changed, args.edge_constraint = psim.Checkbox("Active \nEdgeSpring", args.edge_constraint)
        #
        #         changed, args.deformation_gradient_constraint_wi = psim.InputFloat("wi \nDeformationGradient",
        #                                                                            args.deformation_gradient_constraint_wi)
        #         changed, args.tet_deformation_constraint = psim.Checkbox("Active \nDeformationGradient",
        #                                                                  args.tet_deformation_constraint)
        #
        #         changed, args.strain_limit_constraint_wi = psim.InputFloat("wi \nStrainLimit",
        #                                                                    args.strain_limit_constraint_wi)
        #         changed, args.sigma_min = psim.InputFloat("Minimum singular \nvalue StrainLimit", args.sigma_min)
        #         changed, args.sigma_max = psim.InputFloat("Maximum singular \nvalue StrainLimit", args.sigma_max)
        #
        #         changed, args.tri_strain_constraint = psim.Checkbox("Active \nTriStrain", args.tri_strain_constraint)
        #         changed, args.tet_strain_constraint = psim.Checkbox("Active \nTetStrain", args.tet_strain_constraint)
        #
        #         changed, args.positional_constraint_wi = psim.InputFloat("wi \nPositional constraint",
        #                                                                  args.positional_constraint_wi)
        #
        #         if psim.Button("Apply##Constraints"):
        #             model.immobilize()
        #             model.clear_constraints()
        #             # model.reset_constraints_attributes()
        #             solver.set_dirty()
        #             # ---------------------------------------------------------------------------------------------------
        #
        #             # used for Bar
        #             if args.fix_left_side and not args._fix_left_triggered:
        #                 model.fix_surface_side_vertices(side="left")
        #                 args._fix_left_triggered = True
        #             elif args._fix_left_triggered and not args.fix_left_side:
        #                 model.release_surface_side_vertices(side="left")
        #                 args._fix_left_triggered = False
        #
        #             if args.fix_right_side and not args._fix_right_triggered:
        #                 model.fix_surface_side_vertices(side="right")
        #                 args._fix_right_triggered = True
        #             elif args._fix_right_triggered and not args.fix_right_side:
        #                 model.release_surface_side_vertices(side="right")
        #                 args._fix_right_triggered = False
        #             # ---------------------------------------------------------------------------------------------------
        #
        #             # used for cloth
        #             if args.fix_top_corners and not args._fix_top_corners_triggered:
        #                 model.fix_cloth_corners(side="top")
        #                 args._fix_top_corners_triggered = True
        #             elif args._fix_top_corners_triggered and not args.fix_top_corners:
        #                 model.release_cloth_corners(side="top")
        #                 args._fix_top_corners_triggered = False
        #
        #             if args.fix_bottom_corners and not args._fix_bottom_corners_triggered:
        #                 model.fix_cloth_corners(side="bottom")
        #                 args._fix_bottom_corners_triggered = True
        #             elif args._fix_bottom_corners_triggered and not args.fix_bottom_corners:
        #                 model.release_cloth_corners(side="bottom")
        #                 args._fix_bottom_corners_triggered = False
        #
        #             if args.fix_right_corners and not args._fix_right_corners_triggered:
        #                 model.fix_cloth_corners(side="right")
        #                 args._fix_right_corners_triggered = True
        #             elif args._fix_right_corners_triggered and not args.fix_right_corners:
        #                 model.release_cloth_corners(side="right")
        #                 args._fix_right_corners_triggered = False
        #
        #             if args.fix_left_corners and not args._fix_left_corners_triggered:
        #                 model.fix_cloth_corners(side="left")
        #                 args._fix_left_corners_triggered = True
        #             elif args._fix_left_corners_triggered and not args.fix_left_corners:
        #                 model.release_cloth_corners(side="left")
        #                 args._fix_left_corners_triggered = False
        #             # ---------------------------------------------------------------------------------------------------
        #
        #         psim.BulletText(f"no. Constraints: {len(model.constraints)}")
        #         psim.TreePop()
        #
        #     changed, args.dt = psim.InputFloat("Timestep", args.dt)
        #     changed, args.solver_iterations = psim.InputInt("Solver iterations", args.solver_iterations)
        #     changed, args.mass_per_particle = psim.InputFloat("mass per particle", args.mass_per_particle)
        #     changed, args.is_gravity_active = psim.Checkbox("Gravity", args.is_gravity_active)
        #
        #     changed, args.is_simulating = psim.Checkbox("Simulate", args.is_simulating)
        #
        #
        #
        #     if model is not None:
        #
        #         # # if recording snapshots build output file name/ path
        #         # if record_fom_info:
        #         #     specify_path = ""
        #         #     if model.has_verts_bending_constraints:
        #         #         specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
        #         #         if args.vert_bending_reduced:
        #         #             specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) + "_"
        #         #
        #         #     if model.has_edge_spring_constraints:
        #         #         specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
        #         #         if args.edge_spring_reduced:
        #         #             specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) + "_"
        #         #
        #         #     if model.has_tris_strain_constraints:
        #         #         specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
        #         #         if args.tri_strain_reduced:
        #         #             specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) + "_"
        #         #     if model.has_tets_strain_constraints:
        #         #         specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
        #         #         if args.tet_strain_reduced:
        #         #             specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) + "_"
        #         #     if model.has_tets_deformation_gradient_constraints:
        #         #         specify_path = specify_path + "tets_deformation_gradient_wi" + str(
        #         #             args.deformation_gradient_constraint_wi) + "_"
        #         #         if args.tet_deformation_reduced:
        #         #             specify_path = specify_path + "reduced_" + str(args.tet_deformation_num_components)+ "_"
        #         #
        #         #     output_path += "/" + object_name + "/" + specify_path
        #
        #         # mouse_down_handler = MouseDownHandler(lambda: model.positions.shape[0] > 0, picking_state, solver, physics_params)
        #         # mouse_move_handler = MouseMoveHandler(lambda: model.positions.shape[0] > 0, picking_state, model, lambda: fext)
        #         fext_dragging = mouse_move_handler.fext
        #
        #         pre_draw_handler = PreDrawHandler(lambda: model.positions.shape[0] > 0, args, solver, fext ,
        #                                           record_info=record_fom_info, record_path=output_path)
        #         # print(solver.frame)
        #     if args.is_simulating:
        #         pre_draw_handler.set_animating(True)
        #         pre_draw_handler.handle()

        # Inside interactive_testing_callback
        # io = psim.GetIO()
        # if io.MouseClicked[0]:  # left-click
        #     screen_coords = io.MousePos
        #     current_x, current_y = screen_coords
        #     pick_result = ps.pick(screen_coords=screen_coords)
        #
        #     if pick_result.is_hit and pick_result.structure_name == "model":
        #         # Get modifier
        #         if io.KeyCtrl:
        #             # dragging mode
        #             modifier = "ctrl"
        #         elif io.KeyShift:
        #             # add positional constraint
        #             modifier = "shift"
        #
        #         else:
        #             modifier = None
        #
        #         v_id = pick_result.local_index
        #         pos = pick_result.position
        #
        #         picking_state.vertex = v_id
        #         picking_state.is_picking = (modifier == "ctrl")
        #         picking_state.mouse_x = current_x
        #         picking_state.mouse_y = current_y
        #         print(f"Picked vertex {v_id} at screen {screen_coords} -> position {pos} --> modifier {modifier}")
        #
        #         mouse_down_handler.handle_click(pick_result, button="left", modifier=modifier)
        #     if picking_state.is_picking and mouse_move_handler is not None:
        #
        #         mouse_move_handler.handle_mouse_move()
        #
        # if psim.Button("Cancel Picking"):
        #     picking_state.is_picking = False
        #     model.picked_vert = [False] *len(model.picked_vert )
        #
        # if psim.CollapsingHeader("Visualization"):
        #     changed, wire = psim.Checkbox("Wireframe", ps.get_surface_mesh("mesh").get_edge_width() > 0.0)
        #     if wire:
        #         ps.get_surface_mesh("mesh").set_edge_width(1.0)
        #     else:
        #         ps.get_surface_mesh("mesh").set_edge_width(0.0)
        #     ps.get_surface_mesh("mesh").set_point_radius(psim.InputFloat("Point size", 0.02), relative=True)
        #
        # psim.End()
        psim.End()
    return callback
