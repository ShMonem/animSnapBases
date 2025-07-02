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

bar_width, bar_height, bar_depth = 12, 4, 4
cloth_width, cloth_height = 20, 20
window_open = True  # global state
model = None
mouse_down_handler, mouse_move_handler, pre_draw_handler = None, None, None
physics_params = {
    "is_gravity_active": False,
    "dt": 0.0166667,
    "solver_iterations": 10,
    "mass_per_particle": 10.0,
    "edge_constraint_wi": 1_000_000.0,
    "positional_constraint_wi": 1_000_000_000.0,
    "deformation_gradient_constraint_wi": 10_000_000.0,
    "strain_limit_constraint_wi": 10_000_000.0,
    "sigma_min": 0.99,
    "sigma_max": 1.01,
    "apply_constraints": False,
    "edge_constraint": False,
    "deformation_constraint": False,
    "strain_constraint": False,
    "is_simulating": False,
    "fix_left_side": False,
    "fix_right_side": False,
    "_fix_left_triggered": False,
    "_fix_right_triggered": False,

}

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


def callback():
    global model, solver, fext, bar_width, bar_height, bar_depth,\
        mouse_down_handler, mouse_move_handler, pre_draw_handler,\
        cloth_width, cloth_height, window_open

    psim.PushItemWidth(200)
    psim.TextUnformatted("== Projective Dynamics ==")
    psim.Separator()


    if psim.CollapsingHeader("Geometry"):
        if psim.TreeNode("Bar"):
            changed, bar_width = psim.InputInt("width##Bar", bar_width)
            changed, bar_height = psim.InputInt("height##Bar", bar_height)
            changed, bar_depth = psim.InputInt("depth##Bar", bar_depth)

            if psim.Button("Compute##Bar"):
                V, T, F = get_simple_bar_model(bar_width, bar_height, bar_depth)
                reset_simulation_model(V, F, T, should_rescale=True)
            psim.TreePop()
        if psim.TreeNode("Cloth"):
            changed, cloth_width = psim.InputInt("width##Cloth", cloth_width)
            changed, cloth_height = psim.InputInt("height##Cloth", cloth_height)
            if psim.Button("Compute##Cloth"):
                V, F = get_simple_cloth_model(cloth_width, cloth_height)
                reset_simulation_model(V, F, np.empty((0,3)), should_rescale=True)

            psim.TreePop()

        if model is not None:
            psim.BulletText(f"Vertices: {model.positions.shape[0]}")
            psim.BulletText(f"Triangles: {model.faces.shape[0]}")
            psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

        changed, physics_params["fix_left_side"] = psim.Checkbox("Fix Left\nVertices Side",
                                                                 physics_params["fix_left_side"])
        changed, physics_params["fix_right_side"] = psim.Checkbox("Fix Right\nVertices Side",
                                                                  physics_params["fix_right_side"])

        # One-shot execution logic
        if physics_params["fix_left_side"] and not physics_params["_fix_left_triggered"]:
            model.fix_surface_side_vertices(physics_params, side="left")
            physics_params["_fix_left_triggered"] = True
        elif not physics_params["fix_left_side"]:
            physics_params["_fix_left_triggered"] = False

        if physics_params["fix_right_side"] and not physics_params["_fix_right_triggered"]:
            model.fix_surface_side_vertices(physics_params, side="right")
            physics_params["_fix_right_triggered"] = True
        elif not physics_params["fix_right_side"]:
            physics_params["_fix_right_triggered"] = False


    if psim.CollapsingHeader("Physics"):
        if psim.TreeNode("Constraints"):
            changed, physics_params["edge_constraint_wi"] = psim.InputFloat("wi \nEdgeLength", physics_params["edge_constraint_wi"])
            changed, physics_params["edge_constraint"] = psim.Checkbox("Active \nEdgeLength", physics_params["edge_constraint"])

            changed, physics_params["deformation_gradient_constraint_wi"] = psim.InputFloat("wi \nDeformationGradient", physics_params["deformation_gradient_constraint_wi"])
            changed, physics_params["deformation_constraint"] = psim.Checkbox("Active \nDeformationGradient", physics_params["deformation_constraint"])

            changed, physics_params["strain_limit_constraint_wi"] = psim.InputFloat("wi \nStrainLimit", physics_params["strain_limit_constraint_wi"])
            changed, physics_params["sigma_min"] = psim.InputFloat("Minimum singular \nvalue StrainLimit", physics_params["sigma_min"])
            changed, physics_params["sigma_max"] = psim.InputFloat("Maximum singular \nvalue StrainLimit", physics_params["sigma_max"])
            changed, physics_params["strain_constraint"] = psim.Checkbox("Active \nStrainLimit", physics_params["strain_constraint"])

            changed, physics_params["positional_constraint_wi"] = psim.InputFloat("wi \nPositional constraint", physics_params["positional_constraint_wi"])

            if psim.Button("Apply##Constraints"):
                model.immobilize()
                model.clear_constraints()
                solver.set_dirty()
                if physics_params["edge_constraint"]:
                    model.constrain_edge_lengths(physics_params["edge_constraint_wi"])
                if physics_params["deformation_constraint"]:
                    model.constrain_deformation_gradient(physics_params["deformation_gradient_constraint_wi"])
                if physics_params["strain_constraint"]:
                    model.constrain_strain(
                        physics_params["sigma_min"],
                        physics_params["sigma_max"],
                        physics_params["strain_limit_constraint_wi"])

            psim.BulletText(f"no. Constraints: {len(model.constraints)}")
            psim.TreePop()

        changed, physics_params["dt"] = psim.InputFloat("Timestep", physics_params["dt"])
        changed, physics_params["solver_iterations"] = psim.InputInt("Solver iterations", physics_params["solver_iterations"])
        changed, physics_params["mass_per_particle"] = psim.InputFloat("mass per particle", physics_params["mass_per_particle"])
        changed, physics_params["is_gravity_active"] = psim.Checkbox("Gravity", physics_params["is_gravity_active"])

        changed, physics_params["is_simulating"] = psim.Checkbox("Simulate", physics_params["is_simulating"])

        if model is not None:

            # mouse_down_handler = MouseDownHandler(lambda: model.positions.shape[0] > 0, picking_state, solver, physics_params)
            # mouse_move_handler = MouseMoveHandler(lambda: model.positions.shape[0] > 0, picking_state, model, lambda: fext)
            pre_draw_handler = PreDrawHandler(lambda: model.positions.shape[0] > 0, physics_params, solver, fext)

        if physics_params["is_simulating"]:
            pre_draw_handler.set_animating(True)
            pre_draw_handler.handle()

    if psim.CollapsingHeader("Picking"):
        changed, picking_state["force"] = psim.InputFloat("Dragging force", picking_state["force"])

        # pick_result = ps.pick()
        # if pick_result is not None:
        #     mesh_name, vidx = pick_result
        #     if mesh_name == "model":
        #         picking_state["vertex"] = vidx
        #         picking_state["is_picking"] = True
        #         psim.BulletText(f"Picked vertex: {vidx}")

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