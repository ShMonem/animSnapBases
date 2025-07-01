# main.py

import polyscope as ps
import polyscope.imgui as psim
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

# Placeholder imports (these will be implemented later)
# import geometry.get_simple_bar_model as get_simple_bar_model
# import geometry.get_simple_cloth_model as get_simple_cloth_model
from Constraint_projections import DeformableMesh
from Simulators import Solver
from geometry import get_simple_bar_model, get_simple_cloth_model
# import ui.mouse_down_handler as mouse_down_handler
# import ui.mouse_move_handler as mouse_move_handler
# import ui.physics_params as physics_params
# import ui.picking_state as picking_state
# import ui.pre_draw_handler as pre_draw_handler

# UI state variables
global_ui_state = {
    "show_wireframe": False,
    "point_size": 10.0,
    "bar_width": 12,
    "bar_height": 4,
    "bar_depth": 4,
    "cloth_width": 20,
    "cloth_height": 20,
    "positional_constraint_wi": 1.0,
    "drag_force": 1.0,
    "apply_constraints": False,
    "edge_constraint": False,
    "deformation_constraint": False,
    "strain_constraint": False,
}

model = None
solver_instance = Solver()
fext = None

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
    solver_instance.set_model(model)

    fext = np.zeros_like(model.positions())

    ps.remove_all_structures()
    ps.register_surface_mesh("model", model.positions(), model.faces())

def callback():
    psim.PushItemWidth(200)

    psim.TextUnformatted("== Projective Dynamics ==")
    psim.Separator()

    # File I/O section
    if psim.CollapsingHeader("File I/O", psim.ImGuiTreeNodeFlags_DefaultOpen):
        if psim.Button("Load triangle mesh"):
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(filetypes=[("OBJ or OFF", "*.obj *.off")])
            if filename and os.path.isfile(filename):
                print(f"[INFO] Load triangle mesh: {filename}")
                # TODO: Load and reset model here
        psim.SameLine()
        if psim.Button("Save triangle mesh"):
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.asksaveasfilename(defaultextension=".obj")
            if filename:
                print(f"[INFO] Save triangle mesh to: {filename}")
                # TODO: Save mesh

        if psim.Button("Load tet mesh"):
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(filetypes=[(".mesh", "*.mesh")])
            if filename and os.path.isfile(filename):
                print(f"[INFO] Load tet mesh: {filename}")
                # TODO: Load and reset model here
        psim.SameLine()
        if psim.Button("Save tet mesh"):
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.asksaveasfilename(defaultextension=".mesh")
            if filename:
                print(f"[INFO] Save tet mesh to: {filename}")
                # TODO: Save tet mesh

    # Geometry section
    if psim.TreeNode("Geometry"):
        if psim.Button("Compute Triangle"):
            print("[INFO] Compute Triangle clicked")

        psim.Separator()

        if psim.TreeNode("Bar"):
            changed, global_ui_state["bar_width"] = psim.InputInt("width##Bar", global_ui_state["bar_width"])
            changed, global_ui_state["bar_height"] = psim.InputInt("height##Bar", global_ui_state["bar_height"])
            changed, global_ui_state["bar_depth"] = psim.InputInt("depth##Bar", global_ui_state["bar_depth"])

            if psim.Button("Compute Bar"):
                print("[INFO] Compute Bar clicked")
                V, T, F = get_simple_bar_model(
                    global_ui_state["bar_width"],
                    global_ui_state["bar_height"],
                    global_ui_state["bar_depth"]
                )
                reset_simulation_model(V, F, T, should_rescale=True)

            psim.TreePop()
        if psim.TreeNode("Cloth"):
            changed, global_ui_state["cloth_width"] = psim.InputInt("width##Cloth", global_ui_state["cloth_width"])
            changed, global_ui_state["cloth_height"] = psim.InputInt("height##Cloth", global_ui_state["cloth_height"])

            if psim.Button("Compute Cloth"):
                print("[INFO] Compute Cloth clicked")
                V, F = get_simple_cloth_model(
                    global_ui_state["cloth_width"],
                    global_ui_state["cloth_height"]
                )
                reset_simulation_model(V, F, F, should_rescale=True)

            psim.TreePop()
    # Physics section
    if psim.TreeNode("Physics"):
        if psim.TreeNode("Constraints"):
            changed, global_ui_state["edge_constraint"] = psim.Checkbox("Edge Length Constraint", global_ui_state["edge_constraint"])
            changed, global_ui_state["deformation_constraint"] = psim.Checkbox("Deformation Gradient Constraint", global_ui_state["deformation_constraint"])
            changed, global_ui_state["strain_constraint"] = psim.Checkbox("Strain Constraint", global_ui_state["strain_constraint"])
            changed, global_ui_state["positional_constraint_wi"] = psim.InputFloat("Positional Constraint Weight", global_ui_state["positional_constraint_wi"])
            if psim.Button("Apply Constraints"):
                print("[INFO] Apply Constraints clicked")
            psim.TreePop()
        psim.TreePop()

    # Picking
    if psim.TreeNode("Picking"):
        changed, global_ui_state["drag_force"] = psim.InputFloat("Dragging Force", global_ui_state["drag_force"])
        psim.TreePop()

    # Visualization
    if psim.TreeNode("Visualization"):
        changed, global_ui_state["show_wireframe"] = psim.Checkbox("Wireframe", global_ui_state["show_wireframe"])
        changed, global_ui_state["point_size"] = psim.InputFloat("Point Size", global_ui_state["point_size"])
        psim.TreePop()

    psim.PopItemWidth()

def main():
    ps.init()
    ps.set_user_callback(callback)
    ps.show()

if __name__ == "__main__":
    main()