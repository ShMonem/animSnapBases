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

mouse_down_handler, mouse_move_handler, pre_draw_handler = None, None, None

import config
import argparse

picking_state = {
    "is_picking": False,
    "vertex": 0,
    "mouse_x": 0,
    "mouse_y": 0,
    "force": 400.0,
}

def rescale(V):
    v_mean = np.mean(V, axis=0)
    V -= v_mean
    scale = np.max(V) - np.min(V)
    if scale != 0:
        V /= scale
    return V

def reset_simulation_model(V, F, T, should_rescale=False):
    global model, solver_instance, fext

    if should_rescale:
        V = rescale(V)

    model = DeformableMesh(V, F, T)
    solver.set_model(model)
    fext = np.zeros_like(V)

    ps.remove_all_structures()
    ps.register_surface_mesh("model", model.positions, model.faces, enabled=True)

def main():
    # # ---------------- build parser argument ----------------
    parser = argparse.ArgumentParser()
    
    # Build the system object args holder 
    config.initiate_system_args(parser)

    # Add visualization params
    config.add_visualization_args(parser)

    # Config solver
    config.add_solver_args(parser)

    # Physics parameters
    config.add_physics_args(parser)

    args = parser.parse_args()

    record_fom_info = False
    use_3d_rhs = True
    global output_path
    output_path = "output"
    object_name = ""

    def callback():
        global model, solver, fext, bar_width, bar_height, bar_depth,\
            mouse_down_handler, mouse_move_handler, pre_draw_handler,\
            cloth_width, cloth_height, window_open, object_name, output_path

        psim.PushItemWidth(200)
        psim.TextUnformatted("== Projective Dynamics ==")
        psim.Separator()


        if psim.CollapsingHeader("Geometry"):
            if psim.TreeNode("Bar"):
                
                config.edit_system_args(args, "Bar")

                if psim.Button("Compute##Bar"):
                    V, T, F = get_simple_bar_model(args.bar_width, args.bar_height, args.bar_depth)
                    reset_simulation_model(V, F, T, should_rescale=True)
                    object_name = "Bar"

                psim.TreePop()
            if psim.TreeNode("Cloth"):

                config.edit_system_args(args, "Cloth")

                if psim.Button("Compute##Cloth"):
                    V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
                    reset_simulation_model(V, F, np.empty((0,3)), should_rescale=True)
                    object_name = "Cloth"

                psim.TreePop()

            if model is not None:
                psim.BulletText(f"Vertices: {model.positions.shape[0]}")
                psim.BulletText(f"Triangles: {model.faces.shape[0]}")
                psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
                psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")




             # One-shot execution logic
                    # if args.fix_left_side and not args._fix_left_triggered:
                    #
                    #
                    # if args.fix_right_corners and not args._fix_right_corners_triggered:
                    #     model.fix_cloth_corners( side="right")
                    #     args._fix_right_corners_triggered = True
                    # elif args._fix_right_corners_triggered and not args.fix_right_corners:
                    #     model.release_cloth_corners(side="right")
                    #     args._fix_right_corners_triggered = False
                    #
                    # if args.fix_left_corners and not args._fix_left_corners_triggered:
                    #     model.fix_cloth_corners( side="left")
                    #     args._fix_left_corners_triggered = True
                    # elif args._fix_left_corners_triggered and not args.fix_left_corners:
                    #     model.release_cloth_corners(side="left")
                    #     args._fix_left_corners_triggered = False
                    #
                    # if args.fix_top_corners and not args._fix_top_corners_triggered:
                    #     model.fix_cloth_corners( side="top")
                    #     args._fix_top_corners_triggered = True
                    # elif args._fix_top_corners_triggered and not args.fix_top_corners:
                    #     model.release_cloth_corners(side="top")
                    #     args._fix_top_corners_triggered = False
                    #
                    # if args.fix_bottom_corners and not args._fix_bottom_corners_triggered:
                    #     model.fix_cloth_corners( side="bottom")
                    #     args._fix_bottom_corners_triggered = True
                    # elif args._fix_bottom_corners_triggered and not args.fix_bottom_corners:
                    #     model.release_cloth_corners(side="bottom")
                    #     args._fix_bottom_corners_triggered = False
        if psim.CollapsingHeader("Physics"):
            if psim.TreeNode("Constraints"):

                if object_name == "Bar":
                    changed, args.fix_left_side = psim.Checkbox("Fix Left\nVertices Side", args.fix_left_side)
                    changed, args.fix_right_side = psim.Checkbox("Fix Right\nVertices Side", args.fix_right_side)

                if object_name == "Cloth":
                    # changed, args.fix_left_corners = psim.Checkbox("Fix Left\nCorners Side", args.fix_left_corners)
                    # changed, args.fix_right_corners = psim.Checkbox("Fix Right\nCorners Side", args.fix_right_corners)

                    changed, args.fix_top_corners = psim.Checkbox("Fix Top\nCorners Side", args.fix_top_corners)
                    changed, args.fix_bottom_corners = psim.Checkbox("Fix Bottom\nCorners Side", args.fix_bottom_corners)

                changed, args.vert_bending_constraint_wi = psim.InputFloat("wi \nVertBend", args.vert_bending_constraint_wi)
                changed, args.vert_bending_constraint = psim.Checkbox("Active \nVertBend", args.vert_bending_constraint)

                changed, args.edge_constraint_wi = psim.InputFloat("wi \nEdgeSpring", args.edge_constraint_wi)
                changed, args.edge_constraint = psim.Checkbox("Active \nEdgeSpring", args.edge_constraint)

                changed, args.deformation_gradient_constraint_wi = psim.InputFloat("wi \nDeformationGradient", args.deformation_gradient_constraint_wi)
                changed, args.tet_deformation_constraint = psim.Checkbox("Active \nDeformationGradient", args.tet_deformation_constraint)

                changed, args.strain_limit_constraint_wi = psim.InputFloat("wi \nStrainLimit", args.strain_limit_constraint_wi)
                changed, args.sigma_min = psim.InputFloat("Minimum singular \nvalue StrainLimit", args.sigma_min)
                changed, args.sigma_max = psim.InputFloat("Maximum singular \nvalue StrainLimit", args.sigma_max)

                changed, args.tri_strain_constraint = psim.Checkbox("Active \nTriStrain", args.tri_strain_constraint)
                changed, args.tet_strain_constraint = psim.Checkbox("Active \nTetStrain", args.tet_strain_constraint)

                changed, args.positional_constraint_wi = psim.InputFloat("wi \nPositional constraint", args.positional_constraint_wi)



                if psim.Button("Apply##Constraints"):
                    model.immobilize()
                    model.clear_constraints()
                    model.reset_constraints_attributes()
                    solver.set_dirty()
                    output_path = "output"

                    # used for Bar
                    if args.fix_left_side and not args._fix_left_triggered:
                        model.fix_surface_side_vertices(side="left")
                        args._fix_left_triggered = True
                    elif args._fix_left_triggered and not args.fix_left_side:
                        model.release_surface_side_vertices(side="left")
                        args._fix_left_triggered = False

                    if args.fix_right_side and not args._fix_right_triggered:
                        model.fix_surface_side_vertices( side="right")
                        args._fix_right_triggered = True
                    elif args._fix_right_triggered and not args.fix_right_side:
                        model.release_surface_side_vertices(side="right")
                        args._fix_right_triggered = False

                    # used for cloth
                    if args.fix_top_corners and not args._fix_top_corners_triggered:
                        model.fix_cloth_corners( side="top")
                        args._fix_top_corners_triggered = True
                    elif args._fix_top_corners_triggered and not args.fix_top_corners:
                        model.release_cloth_corners(side="top")
                        args._fix_top_corners_triggered = False

                    if args.fix_bottom_corners and not args._fix_bottom_corners_triggered:
                        model.fix_cloth_corners( side="bottom")
                        args._fix_bottom_corners_triggered = True
                    elif args._fix_bottom_corners_triggered and not args.fix_bottom_corners:
                        model.release_cloth_corners(side="bottom")
                        args._fix_bottom_corners_triggered = False

                    if args.vert_bending_constraint:
                        model.add_vertex_bending_constraint(args.vert_bending_constraint_wi)
                    if args.edge_constraint:
                        model.add_edge_spring_constrain(args.edge_constraint_wi)

                    if args.tri_strain_constraint:
                        model.add_tri_constrain_strain(
                            args.sigma_min,
                            args.sigma_max,
                            args.strain_limit_constraint_wi)

                    if args.tet_deformation_constraint:
                        model.add_tet_constrain_deformation_gradient(args.deformation_gradient_constraint_wi)
                    if args.tet_strain_constraint:
                        model.add_tet_constrain_strain(
                            args.sigma_min,
                            args.sigma_max,
                            args.strain_limit_constraint_wi)

                    # if recording snapshots build output file name/ path
                    if record_fom_info:

                        specify_path = ""
                        if model.has_verts_bending_constraints:
                            specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi)

                        if model.has_edge_spring_constraints:
                            specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi)

                        if model.has_tris_strain_constraints:
                            specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi)

                        if model.has_tets_strain_constraints:
                            specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi)

                        if model.has_tets_deformation_gradient_constraints:
                            specify_path = specify_path + "tets_deformation_gradient_wi" + str(
                                args.deformation_gradient_constraint_wi)

                        output_path += "/" + object_name + "/" + specify_path


                psim.BulletText(f"no. Constraints: {len(model.constraints)}")
                psim.TreePop()

            changed, args.dt = psim.InputFloat("Timestep", args.dt)
            changed, args.solver_iterations = psim.InputInt("Solver iterations", args.solver_iterations)
            changed, args.mass_per_particle = psim.InputFloat("mass per particle", args.mass_per_particle)
            changed, args.is_gravity_active = psim.Checkbox("Gravity", args.is_gravity_active)

            changed, args.is_simulating = psim.Checkbox("Simulate", args.is_simulating)


            if model is not None:
                # mouse_down_handler = MouseDownHandler(lambda: model.positions.shape[0] > 0, picking_state, solver, physics_params)
                # mouse_move_handler = MouseMoveHandler(lambda: model.positions.shape[0] > 0, picking_state, model, lambda: fext)
                pre_draw_handler = PreDrawHandler(lambda: model.positions.shape[0] > 0, args, solver, fext, record_info=record_fom_info, record_path= output_path)

            if args.is_simulating:
                pre_draw_handler.set_animating(True)
                pre_draw_handler.handle()

        if psim.CollapsingHeader("Picking"):
            changed, picking_state["force"] = psim.InputFloat("Dragging force", picking_state["force"])

            pick_result = ps.pick()
            if pick_result is not None:
                meshindices_name, vidx = pick_result
                if mesh_name == "model":
                    picking_state["vertex"] = vidx
                    picking_state["is_picking"] = True
                    psim.BulletText(f"Picked vertex: {vidx}")

        if psim.CollapsingHeader("Visualization"):
            changed, wire = psim.Checkbox("Wireframe", ps.get_surface_mesh("mesh").get_edge_width() > 0.0)
            if wire:
                ps.get_surface_mesh("mesh").set_edge_width(1.0)
            else:
                ps.get_surface_mesh("mesh").set_edge_width(0.0)
            ps.get_surface_mesh("mesh").set_point_radius(psim.InputFloat("Point size", 0.02), relative=True)

        psim.End()

    # Register callback
    ps.init()
    ps.set_user_callback(callback)

    # Launch viewer
    ps.show()


if __name__ == '__main__':
    main()