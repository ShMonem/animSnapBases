import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import KDTree
import config
from Constraint_projections import DeformableMesh
from geometry import get_simple_bar_model, get_simple_cloth_model, get_simple_bar_model_with_surface_points_only, compute_lumped_mass_matrix
from usr_interface import MouseDownHandler, MouseMoveHandler, PreDrawHandler, PickingState

from Simulators import animSnapBasesSolver, Solver
import trimesh
import meshio
from utils import check_dir_exists

# declare global variables
model = None
fext = None
solver = None

picking_state = PickingState()
mouse_down_handler = None
mouse_move_handler = None

# picking_state = {
#     "is_picking": False,
#     "vertex": 0,
#     "mouse_x": 0,
#     "mouse_y": 0,
#     "force": 400.0,
# }

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
    elif args.solver == "Solver":
        return Solver()
    else:
        raise ValueError("Unknown solver name")

def rescale(V):
    v_mean = np.mean(V, axis=0)
    V -= v_mean
    scale = np.max(V) - np.min(V)
    if scale != 0:
        V /= scale
    return V

def reset_simulation_model(V, F, T, should_rescale=False, params=None):
    global model, fext, solver
    if should_rescale:
        V = rescale(V)

    model = DeformableMesh(V, F, T)
    solver.set_model(model)
    fext = np.zeros_like(V)

    ps.remove_all_structures()
    ps.register_surface_mesh("model", model.positions, model.faces, enabled=True)
    ps.get_surface_mesh("model").set_selection_mode("vertices_only")  # or "vertex_only"

    ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
    # Set camera to look down from above, along negative Z
    ps.look_at(
        target=(0.0, 0.0, 0.0),  # Look at the origin (the floor)
        camera_location=(0.0, 0.0, 3.0)  # Camera is 3 units above, looking down
    )

# Example callbacks called in main.py
def bar_automated_deformationgradient_callback(args, record_fom_info = False, params=None,experiment="bar_automated_deformationgradient"):
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

            params.edit_system_args(args, "Bar")

            V, T, F, _ = get_simple_bar_model(args.bar_width, args.bar_height, args.bar_depth)

            reset_simulation_model(V, F, T, should_rescale=True)
            object_name = "bar"

            check_dir_exists(os.path.join(output_path, object_name))
            mesh = meshio.Mesh(
                points=V,
                cells=[
                    ("triangle", F),
                    ("tetra", T)
                ]
            )
            mesh.write(os.path.join(output_path, object_name, object_name+".mesh"))

            mesh_surface = trimesh.Trimesh(vertices=V, faces=F)
            mesh_surface.export(os.path.join(output_path, object_name, object_name + ".obj"))

            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            model.fix_surface_side_vertices(side="left")
            model.fix_surface_side_vertices(side="right")

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
            if args.tet_strain_constraint:
                model.add_tet_constrain_strain(args.sigma_min, args.sigma_max, args.strain_limit_constraint_wi)
            if args.tet_deformation_constraint:
                model.add_tet_constrain_deformation_gradient(args.deformation_gradient_constraint_wi)
            # if recording snapshots build output file name/ path
            if record_fom_info:
                constrproj_case = "constraint_projection/FOM"
                if solver.has_reduced_constraint_projectios:
                    constrproj_case = "constraint_projection/" + args.constraint_projection_basis_type

                specify_path = ""
                if model.has_verts_bending_constraints:
                    specify_path = specify_path + "verts_bending_wi" + str(args.vert_bending_constraint_wi) + "_"
                    if args.vert_bending_reduced :
                        specify_path = specify_path + "reduced_" + str(args.vert_bending_num_components) +"_"

                if model.has_edge_spring_constraints:
                    specify_path = specify_path + "edge_spring_wi" + str(args.edge_constraint_wi) + "_"
                    if args.edge_spring_reduced :
                        specify_path = specify_path + "reduced_" + str(args.edge_spring_num_components) +"_"

                if model.has_tris_strain_constraints:
                    specify_path = specify_path + "tris_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tri_strain_reduced :
                        specify_path = specify_path + "reduced_" + str(args.tri_strain_num_components) +"_"
                if model.has_tets_strain_constraints:
                    specify_path = specify_path + "tets_strain_wi" + str(args.strain_limit_constraint_wi) + "_"
                    if args.tet_strain_reduced :
                        specify_path = specify_path + "reduced_" + str(args.tet_strain_num_components) +"_"
                if model.has_tets_deformation_gradient_constraints:
                    specify_path = specify_path + "tets_deformation_gradient_wi" + str(args.deformation_gradient_constraint_wi) + "_"
                    if args.tet_deformation_reduced :
                        specify_path = specify_path + "reduced_"+ str(args.tet_deformation_num_components)+"_"


                output_path += "/" + object_name + "/" + experiment + "/" + "/" + constrproj_case + "/" + specify_path + "/"
                check_dir_exists(output_path)

                solver.set_record_path(output_path)
                solver.set_store_p(record_fom_info)
            solver.set_dirty()

        elif solver.frame == 40:
            print("Frame 10: Releasing left side")
            model.release_surface_side_vertices(side="left")


        elif solver.frame == 80:
            print("Frame 10: Releasing right side")
            model.release_surface_side_vertices(side="right")
        #
        # elif solver.frame == 140:
        #     print("Frame 30: Releasing all corners")
        #     model.release_cloth_corners(side="top")
        #     model.release_cloth_corners(side="bottom")
        #     model.fix_cloth_corners(side="right")


        elif solver.frame == 144:
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
            mesh = trimesh.Trimesh(vertices=V, faces=F)
            mesh.export(os.path.join(output_path, object_name, object_name+".obj"))

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
            mesh = trimesh.Trimesh(vertices=V, faces=F)
            mesh.export(os.path.join(output_path, object_name, object_name+".obj"))

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
            mesh = trimesh.Trimesh(vertices=V, faces=F)
            mesh.export(os.path.join(output_path, object_name, object_name+".obj"))

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


def cloth_snapshots(args, record_fom_info = False, params=None,experiment="cloth_automated_snapshots"):
    global model, fext, solver
    solver = get_solver_class_from_name(args)
    is_simulating = True
    output_path = args.output_dir


    poking_frames_per_point = 20
    rest_frames_per_point = 10
    poking_series = None
    poked_points = None
    number_pockes = 15
    total_frames = number_pockes*(rest_frames_per_point + poking_frames_per_point)

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

        # Sample seeds using farthest point sampling
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

    def callback():
        nonlocal output_path, is_simulating, poking_series, poked_points, poking_frames_per_point, rest_frames_per_point, number_pockes, total_frames
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
            mesh = trimesh.Trimesh(vertices=V, faces=F)
            mesh.export(os.path.join(output_path, object_name, object_name + ".obj"))

            psim.PushItemWidth(200)
            psim.TextUnformatted("== Projective Dynamics ==")
            psim.Separator()

            model.fix_surface_side_vertices(side="top")
            # model.fix_cloth_corners(side="bottom")

            # find the closest vertex to center

            # Generate motion serise for poking
            # Generate z values: 0 → 1 ->> -1 linearly
            poking_series = create_poke_z_motion_with_jumps(poking_frames_per_point, rest_frames_per_point, number_pockes, z_range=0.2)
            poked_points, labels = get_voronoi_seeds_and_partition(V, F, number_pockes)

            # Apply any desired constraints
            model.immobilize()
            model.clear_constraints()
            model.reset_constraints_attributes()

            model.add_positional_constraint(poked_points[0], args.positional_constraint_wi,
                                            motion_type="user_defined", frame_shift=poking_series)
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

        elif solver.frame % (poking_frames_per_point+rest_frames_per_point) == 0.0 and solver.frame > 0:
            i = solver.frame // (poking_frames_per_point+rest_frames_per_point)
            if i <= number_pockes:
                model.add_positional_constraint(poked_points[i], args.positional_constraint_wi,
                                                motion_type="user_defined", frame_shift=poking_series)
                model.picked_vert[poked_points[i]] = True
                solver.set_dirty()
                print(f"Poking - positional constraint added to {i} vertex")

        elif solver.frame % (poking_frames_per_point+rest_frames_per_point) == poking_frames_per_point and solver.frame > 0:
            i = solver.frame // (poking_frames_per_point + rest_frames_per_point)
            if i <= number_pockes:
                print(f"Removing - positional constraint remover from {i} vertex")
                model.remove_positional_constraint(poked_points[i])
                model.picked_vert[poked_points[i]] = False
                solver.set_dirty()


        if solver.frame == total_frames:
            model.release_surface_side_vertices(side="top")

        if solver.frame == total_frames + rest_frames_per_point:
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

def interacrive_testing_callback(args, record_fom_info = False, params=None):
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
        if psim.CollapsingHeader("Geometry"):
            if psim.TreeNode("Bar"):

                params.edit_system_args(args, "Bar")

                if psim.Button("Compute##Bar"):
                    V, T, F, _ = get_simple_bar_model(args.bar_width, args.bar_height, args.bar_depth)
                    reset_simulation_model(V, F, T, should_rescale=True)
                    object_name = "Bar"

                psim.TreePop()
            if psim.TreeNode("Cloth"):

                params.edit_system_args(args, "Cloth")

                if psim.Button("Compute##Cloth"):
                    V, F = get_simple_cloth_model(args.cloth_width, args.cloth_height)
                    reset_simulation_model(V, F, np.empty((0, 3)), should_rescale=True)
                    object_name = "Cloth"

                psim.TreePop()

            if model is not None:
                set_up_mouse_handler(args, model, fext)


                psim.BulletText(f"Vertices: {model.positions.shape[0]}")
                psim.BulletText(f"Triangles: {model.faces.shape[0]}")
                psim.BulletText(f"Edges: {model.count_edges(model.faces)}")
                psim.BulletText(f"Tetrahedrons: {model.elements.shape[0]}")

        if psim.CollapsingHeader("Physics"):
            if psim.TreeNode("Constraints"):

                if object_name == "Bar":
                    changed, args.fix_left_side = psim.Checkbox("Fix Left\nVertices Side", args.fix_left_side)
                    changed, args.fix_right_side = psim.Checkbox("Fix Right\nVertices Side", args.fix_right_side)

                if object_name == "Cloth":
                    changed, args.fix_left_corners = psim.Checkbox("Fix Left\nCorners Side", args.fix_left_corners)
                    changed, args.fix_right_corners = psim.Checkbox("Fix Right\nCorners Side", args.fix_right_corners)

                    changed, args.fix_top_corners = psim.Checkbox("Fix Top\nCorners Side", args.fix_top_corners)
                    changed, args.fix_bottom_corners = psim.Checkbox("Fix Bottom\nCorners Side",
                                                                     args.fix_bottom_corners)

                changed, args.vert_bending_constraint_wi = psim.InputFloat("wi \nVertBend",
                                                                           args.vert_bending_constraint_wi)
                changed, args.vert_bending_constraint = psim.Checkbox("Active \nVertBend", args.vert_bending_constraint)

                changed, args.edge_constraint_wi = psim.InputFloat("wi \nEdgeSpring", args.edge_constraint_wi)
                changed, args.edge_constraint = psim.Checkbox("Active \nEdgeSpring", args.edge_constraint)

                changed, args.deformation_gradient_constraint_wi = psim.InputFloat("wi \nDeformationGradient",
                                                                                   args.deformation_gradient_constraint_wi)
                changed, args.tet_deformation_constraint = psim.Checkbox("Active \nDeformationGradient",
                                                                         args.tet_deformation_constraint)

                changed, args.strain_limit_constraint_wi = psim.InputFloat("wi \nStrainLimit",
                                                                           args.strain_limit_constraint_wi)
                changed, args.sigma_min = psim.InputFloat("Minimum singular \nvalue StrainLimit", args.sigma_min)
                changed, args.sigma_max = psim.InputFloat("Maximum singular \nvalue StrainLimit", args.sigma_max)

                changed, args.tri_strain_constraint = psim.Checkbox("Active \nTriStrain", args.tri_strain_constraint)
                changed, args.tet_strain_constraint = psim.Checkbox("Active \nTetStrain", args.tet_strain_constraint)

                changed, args.positional_constraint_wi = psim.InputFloat("wi \nPositional constraint",
                                                                         args.positional_constraint_wi)

                if psim.Button("Apply##Constraints"):
                    model.immobilize()
                    model.clear_constraints()
                    model.reset_constraints_attributes()
                    solver.set_dirty()
                    # ---------------------------------------------------------------------------------------------------

                    # used for Bar
                    if args.fix_left_side and not args._fix_left_triggered:
                        model.fix_surface_side_vertices(side="left", args=args)
                        args._fix_left_triggered = True
                    elif args._fix_left_triggered and not args.fix_left_side:
                        model.release_surface_side_vertices(side="left")
                        args._fix_left_triggered = False

                    if args.fix_right_side and not args._fix_right_triggered:
                        model.fix_surface_side_vertices(side="right", args=args)
                        args._fix_right_triggered = True
                    elif args._fix_right_triggered and not args.fix_right_side:
                        model.release_surface_side_vertices(side="right")
                        args._fix_right_triggered = False
                    # ---------------------------------------------------------------------------------------------------

                    # used for cloth
                    if args.fix_top_corners and not args._fix_top_corners_triggered:
                        model.fix_cloth_corners(side="top")
                        args._fix_top_corners_triggered = True
                    elif args._fix_top_corners_triggered and not args.fix_top_corners:
                        model.release_cloth_corners(side="top")
                        args._fix_top_corners_triggered = False

                    if args.fix_bottom_corners and not args._fix_bottom_corners_triggered:
                        model.fix_cloth_corners(side="bottom")
                        args._fix_bottom_corners_triggered = True
                    elif args._fix_bottom_corners_triggered and not args.fix_bottom_corners:
                        model.release_cloth_corners(side="bottom")
                        args._fix_bottom_corners_triggered = False

                    if args.fix_right_corners and not args._fix_right_corners_triggered:
                        model.fix_cloth_corners(side="right")
                        args._fix_right_corners_triggered = True
                    elif args._fix_right_corners_triggered and not args.fix_right_corners:
                        model.release_cloth_corners(side="right")
                        args._fix_right_corners_triggered = False

                    if args.fix_left_corners and not args._fix_left_corners_triggered:
                        model.fix_cloth_corners(side="left")
                        args._fix_left_corners_triggered = True
                    elif args._fix_left_corners_triggered and not args.fix_left_corners:
                        model.release_cloth_corners(side="left")
                        args._fix_left_corners_triggered = False
                    # ---------------------------------------------------------------------------------------------------
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

                psim.BulletText(f"no. Constraints: {len(model.constraints)}")
                psim.TreePop()

            changed, args.dt = psim.InputFloat("Timestep", args.dt)
            changed, args.solver_iterations = psim.InputInt("Solver iterations", args.solver_iterations)
            changed, args.mass_per_particle = psim.InputFloat("mass per particle", args.mass_per_particle)
            changed, args.is_gravity_active = psim.Checkbox("Gravity", args.is_gravity_active)

            changed, args.is_simulating = psim.Checkbox("Simulate", args.is_simulating)

            if model is not None:

                # if recording snapshots build output file name/ path
                if record_fom_info:
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

                    output_path += "/" + object_name + "/" + specify_path

                # mouse_down_handler = MouseDownHandler(lambda: model.positions.shape[0] > 0, picking_state, solver, physics_params)
                # mouse_move_handler = MouseMoveHandler(lambda: model.positions.shape[0] > 0, picking_state, model, lambda: fext)
                fext_dragging = mouse_move_handler.fext

                pre_draw_handler = PreDrawHandler(lambda: model.positions.shape[0] > 0, args, solver, fext + fext_dragging,
                                                  record_info=record_fom_info, record_path=output_path)
                # print(solver.frame)
            if args.is_simulating:
                pre_draw_handler.set_animating(True)
                pre_draw_handler.handle()

        # Inside interactive_testing_callback
        io = psim.GetIO()
        if io.MouseClicked[0]:  # left-click
            screen_coords = io.MousePos
            current_x, current_y = screen_coords
            pick_result = ps.pick(screen_coords=screen_coords)

            if pick_result.is_hit and pick_result.structure_name == "model":
                # Get modifier
                if io.KeyCtrl:
                    # dragging mode
                    modifier = "ctrl"
                elif io.KeyShift:
                    # add positional constraint
                    modifier = "shift"

                else:
                    modifier = None

                v_id = pick_result.local_index
                pos = pick_result.position

                picking_state.vertex = v_id
                picking_state.is_picking = (modifier == "ctrl")
                picking_state.mouse_x = current_x
                picking_state.mouse_y = current_y
                print(f"Picked vertex {v_id} at screen {screen_coords} -> position {pos} --> modifier {modifier}")

                mouse_down_handler.handle_click(pick_result, button="left", modifier=modifier)
            if picking_state.is_picking and mouse_move_handler is not None:

                mouse_move_handler.handle_mouse_move()

        if psim.Button("Cancel Picking"):
            picking_state.is_picking = False
            model.picked_vert = [False] *len(model.picked_vert )

        if psim.CollapsingHeader("Visualization"):
            changed, wire = psim.Checkbox("Wireframe", ps.get_surface_mesh("mesh").get_edge_width() > 0.0)
            if wire:
                ps.get_surface_mesh("mesh").set_edge_width(1.0)
            else:
                ps.get_surface_mesh("mesh").set_edge_width(0.0)
            ps.get_surface_mesh("mesh").set_point_radius(psim.InputFloat("Point size", 0.02), relative=True)

        psim.End()

    return callback
