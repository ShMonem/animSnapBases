
import polyscope as ps
import polyscope.imgui as psim
import numpy as np

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
        self.force = 400.0
        self.mouse_x = 0
        self.mouse_y = 0

class MouseDownHandler:
    def __init__(self, is_model_ready, picking_state, solver, args):
        self.is_model_ready = is_model_ready
        self.picking_state = picking_state
        self.solver = solver
        self.physics_params = args

    def handle_click(self, button: str, modifier: str):
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

        model = self.solver.model()
        if model is None:
            return False

        # Polyscope picking
        pick_result = ps.pick_vertex("mesh")  # Assuming "mesh" is the registered name
        if pick_result is None:
            return False

        clicked_v_id, picked_pos = pick_result
        if clicked_v_id is None:
            return False

        self.picking_state.vertex = clicked_v_id
        self.picking_state.is_picking = (modifier == "ctrl")

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

    def handle_mouse_move(self, mouse_x, mouse_y):
        """
        Called when mouse moves. Applies force if dragging.
        """
        if not self.is_model_ready() or not self.picking_state.is_picking:
            return False

        v_id = self.picking_state.vertex
        pos0 = self.model.positions[v_id]  # initial vertex pos

        # Estimate force direction: simple screen delta projected into world space
        dx = mouse_x - self.picking_state.mouse_x
        dy = mouse_y - self.picking_state.mouse_y

        # Use camera right & up as basis (simplified interaction model)
        cam = ps.get_camera_state()
        cam_dir = np.array(cam["lookDirection"])
        cam_up = np.array(cam["upDirection"])
        cam_right = np.cross(cam_dir, cam_up)
        cam_right /= np.linalg.norm(cam_right)

        cam_up /= np.linalg.norm(cam_up)

        move_3d = dx * cam_right + dy * cam_up
        direction = move_3d / np.linalg.norm(move_3d + 1e-10)

        # Apply external force
        self.fext[v_id] = direction * self.picking_state.force

        # Update mouse state
        self.picking_state.mouse_x = mouse_x
        self.picking_state.mouse_y = mouse_y

        # Optionally show a point on the mesh
        ps.remove_point_cloud("picked_point")
        ps.register_point_cloud("picked_point", self.model.positions[[v_id]])
        ps.get_point_cloud("picked_point").set_color((1.0, 0.0, 0.0))
        ps.get_point_cloud("picked_point").set_point_radius(0.01, relative=True)

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

            self.solver.step(self.fext, self.physics_params.solver_iterations,
                             store_stacked_projections=self.record_info, record_path=self.record_path)

            # Reset fext and update mesh
            self.fext[:] = 0.0

            if ps.has_surface_mesh("model"):
                ps.remove_surface_mesh("model")
            ps.register_surface_mesh("model", model.positions, model.faces, color=(0.4, 0.4, 0.9))

        # -- 3. Show fixed points
        fixed_indices = [i for i, fix in enumerate(model.get_fixed_indices()) if fix]
        if fixed_indices:
            #ps.remove_point_cloud("fixed_points")
            fixed_positions = model.positions[fixed_indices]
            ps.register_point_cloud("fixed_points", fixed_positions, color=(1.0, 0.0, 0.0))
            ps.get_point_cloud("fixed_points").set_radius(0.01, relative=True)
