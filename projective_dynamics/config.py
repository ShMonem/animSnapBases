
import json
import os

def load_system_params(json_path="config.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    return data

system_params = load_system_params()

def initiate_system_args(parser):
    parser.add_argument("--system_name", type=str, default="not_yet_picked")
    parser.add_argument("--bar_width", type=str, default= 0)
    parser.add_argument("--bar_height", type=str, default= 0)
    parser.add_argument("--bar_depth", type=str, default= 0)
    parser.add_argument("--cloth_width", type=str, default= 0)
    parser.add_argument("--cloth_height", type=str, default= 0)


def edit_system_args(args, system_name):
    args.system_name = system_name
    
    if system_name == "Bar":
        args.bar_width = system_params["system"][system_name]["bar_width"]
        args.bar_height = system_params["system"][system_name]["bar_height"]
        args.bar_depth = system_params["system"][system_name]["bar_depth"]

    elif system_name == "Cloth":
        args.cloth_width =system_params["system"][system_name]["cloth_width"]
        args.cloth_height = system_params["system"][system_name]["cloth_height"]

def add_visualization_args(parser):
    parser.add_argument("--window_open", type=str, default=system_params["visualization_params"]["window_open"])
    parser.add_argument("--is_simulating", type=str, default=system_params["visualization_params"]["is_simulating"])

def add_solver_args(parser):
    solver = system_params["solver_params"]
    parser.add_argument("--solver_window_open", type=bool, default=solver["window_open"])
    parser.add_argument("--solver_is_simulating", type=bool, default=solver["is_simulating"])
    parser.add_argument("--dt", type=float, default=solver["dt"])
    parser.add_argument("--solver_iterations", type=int, default=solver["solver_iterations"])


def add_physics_args(parser):
    physics = system_params["physics_params"]

    parser.add_argument("--mass_per_particle", type=float, default=physics["mass_per_particle"])

    parser.add_argument("--vert_bending_constraint", type=float, default=physics["vert_bending_constraint"])
    parser.add_argument("--vert_bending_constraint_wi", type=float, default=physics["vert_bending_constraint_wi"])

    parser.add_argument("--edge_constraint_wi", type=float, default=physics["edge_constraint_wi"])
    parser.add_argument("--positional_constraint_wi", type=float, default=physics["positional_constraint_wi"])
    parser.add_argument("--deformation_gradient_constraint_wi", type=float,
                        default=physics["deformation_gradient_constraint_wi"])
    parser.add_argument("--strain_limit_constraint_wi", type=float, default=physics["strain_limit_constraint_wi"])
    parser.add_argument("--sigma_min", type=float, default=physics["sigma_min"])
    parser.add_argument("--sigma_max", type=float, default=physics["sigma_max"])

    parser.add_argument("--apply_constraints", type=bool, default=physics["apply_constraints"])
    parser.add_argument("--tri_strain_constraint", type=bool, default=physics["tri_strain_constraint"])

    parser.add_argument("--edge_constraint", type=bool, default=physics["edge_constraint"])
    parser.add_argument("--tet_deformation_constraint", type=bool, default=physics["tet_deformation_constraint"])
    parser.add_argument("--tet_strain_constraint", type=bool, default=physics["tet_strain_constraint"])

    parser.add_argument("--is_gravity_active", type=bool, default=physics["is_gravity_active"])
    parser.add_argument("--fix_left_side", type=bool, default=physics["fix_left_side"])
    parser.add_argument("--fix_right_side", type=bool, default=physics["fix_right_side"])
    parser.add_argument("--_fix_left_triggered", type=bool, default=physics["_fix_left_triggered"])
    parser.add_argument("--_fix_right_triggered", type=bool, default=physics["_fix_right_triggered"])



        
