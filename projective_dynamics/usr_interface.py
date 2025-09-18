import os.path

import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import pygame
import os
pygame.init()
info = pygame.display.Info()
class PhysicsParams:
    def __init__(self):
        self.is_gravity_active = False
        self.dt = 0.0166667
        self.solver_iterations = 10
        self.mass_per_particle = 10.0
        self.edge_constraint_wi = 1_000_000.0
        self.positional_constraint_wi = 1_000_000_000.0
        self.deformation_gradient_constraint_wi = 10_000_000.0
        self.strain_limit_constraint_wi = 10_000_000.0

class PickingState:
    def __init__(self):
        self.is_picking = False
        self.vertex = 0
        self.force = 10000.0
        self.mouse_x = 0
        self.mouse_y = 0

class MouseDownHandler:
    def __init__(self, is_model_ready, picking_state, solver, args):
        self.is_model_ready = is_model_ready
        self.picking_state = picking_state
        self.solver = solver
        self.physics_params = args

    def handle_click(self, pick_result, button: str, modifier: str):
        """
        Handles a click in the UI or triggered event.
        Requires that a vertex has been picked in Polyscope.
        :param button: 'left', 'right', etc.
        :param modifier: 'ctrl', 'shift', or None
        """
        if not self.is_model_ready():
            return False

        if button != "left":
            return False

        model = self.solver.model
        if model is None:
            return False

        # Polyscope picking
        # pick_result = ps.pick_vertex("mesh")  # Assuming "mesh" is the registered name
        if pick_result is None:
            return False

        clicked_v_id, picked_pos = pick_result.local_index, pick_result.position
        if clicked_v_id is None:
            return False

        self.picking_state.vertex = clicked_v_id
        self.picking_state.is_picking = (modifier == "ctrl")

        if self.picking_state.is_picking:
            model.toggle_picked(clicked_v_id)

        if modifier == "shift":
            model.toggle_fixed(clicked_v_id, self.physics_params.mass_per_particle)
            model.add_positional_constraint(
                clicked_v_id, self.physics_params.positional_constraint_wi
            )
            self.solver.set_dirty()

        return True

class MouseMoveHandler:
    def __init__(self, is_model_ready, picking_state, model, fext):
        self.is_model_ready = is_model_ready
        self.picking_state = picking_state
        self.model = model
        self.fext = fext

    def handle_mouse_move(self):
        if not self.is_model_ready() or not self.picking_state.is_picking:
            return False

        v_id = self.picking_state.vertex

        # Estimate direction in screen-space

        # 1. Get viewport dimensions
        # viewport_width = info.current_w
        viewport_height = info.current_h

        # 2. Get current and previous screen coordinates
        x1 = self.picking_state.mouse_x
        y1 = viewport_height - self.picking_state.mouse_y

        io = psim.GetIO()
        x2 = io.MousePos[0]
        y2 = viewport_height - io.MousePos[1]

        # 4. Unproject 2D -> 3D
        p1 = np.array([x1, y1, 0.5])
        p2 = np.array([x2, y2, 0.5])
        print(p1, p2)
        # Map 2D delta to 3D dragging direction arbitrarily
        # For example: use upward dragging to apply force along +Y, right dragging to +X
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        else:
            direction[:] = 0.0

        self.fext[v_id] += direction * self.picking_state.force

        # Update stored mouse coords
        self.picking_state.mouse_x = io.MousePos[0]
        self.picking_state.mouse_y = io.MousePos[1]

        # Visual feedback
        # ps.remove_point_cloud("picked_point")
        # ps.register_point_cloud("picked_point", self.model.positions[[v_id]])
        # ps.get_point_cloud("picked_point").set_color((1.0, 1.0, 0.0))
        # ps.get_point_cloud("picked_point").set_radius(0.01, relative=True)

        return True


class PreDrawHandler:
    def __init__(self, is_model_ready, args, solver, fext, record_info=False, record_path=None):
        self.is_model_ready = is_model_ready
        self.physics_params = args
        self.solver = solver
        self.fext = fext
        self._animating = False  # simulate viewer.core().is_animating
        self.record_info = record_info
        self.record_path = record_path

    def set_animating(self, flag: bool):
        self._animating = flag

    def handle(self):
        if not self.is_model_ready():
            return

        model = self.solver.model
        mass_value = float(self.physics_params.mass_per_particle)

        # -- 1. Update mass
        for i in range(model.mass.shape[0]):
            if model.is_fixed(i):
                continue
            if not np.isclose(model.mass[i], mass_value, atol=1e-5):
                model.mass[i] = mass_value
                self.solver.set_dirty()

        # -- 2. Apply gravity & simulate if animating
        if self._animating:
            gravity = 9.81 if self.physics_params.is_gravity_active else 0.0
            self.fext[:, 1] -= gravity * self.physics_params.mass_per_particle

            if not self.solver.ready():
                self.solver.prepare(self.physics_params, store_fom_info=self.record_info, record_path=self.record_path)

            self.solver.step(self.fext, self.physics_params.solver_iterations)

            # Reset fext and update mesh
            self.fext[:] = 0.0

            if self.solver.has_reduced_constraint_projectios:
                color = (64 / 255, 224 / 255, 208 / 255)  # turquoise
            else:
                color = (0.4, 0.4, 0.9)  # light_purple

            if ps.has_surface_mesh("model"):
                ps.remove_surface_mesh("model")

            ps.register_surface_mesh("model", model.positions, model.faces, color=color, edge_width=1.0)

            # update_camera_to_mesh_center(model)
            ps.reset_camera_to_home_view()

            if self.record_info:
                filename = os.path.join(self.record_path, "frame"+str(self.solver.frame)+".png")
                ps.screenshot(filename, transparent_bg=True)

        # -- 3. Show fixed points
        fixed_indices = [i for i, fix in enumerate(model.get_fixed_indices()) if fix]
        picked_indices = [i for i, pick in enumerate(model.get_picked_verts()) if pick]


        if fixed_indices:
            fixed_positions = model.positions[fixed_indices]
            ps.register_point_cloud("fixed_points", fixed_positions, color=(1.0, 0.0, 0.0))
            # ps.register_point_cloud("fixed_points", model.positions[0:2], color=(1.0, 1.0, 0.0))

            ps.get_point_cloud("fixed_points").set_radius(0.01, relative=True)

        if picked_indices:
            picked_positions = model.positions[picked_indices]
            ps.register_point_cloud("picked_points", picked_positions, color=(1.0, 0.0, 1.0))
            ps.get_point_cloud("picked_points").set_radius(0.01, relative=True)
        else:
            if ps.has_point_cloud("picked_points"):
                ps.remove_point_cloud("picked_points")




def update_camera_to_mesh_center(model, distance=3.0, direction=np.array([0, 0, 1])):
    if model.positions is not None:
        center = np.mean(model.positions, axis=0)
        eye = center + direction * distance  # place the camera some distance away
        ps.look_at(camera_location=eye, target=center)
