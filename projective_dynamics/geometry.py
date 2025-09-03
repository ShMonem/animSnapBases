# pd/geometry/get_simple_bar_model.py

import numpy as np
import igl


def generate_surface_mesh(width, height, depth):
    """
    Generate a surface mesh (vertices and faces) of a cuboid grid.

    Returns:
    - V_surface: (N, 3) vertex positions on the surface
    - F: (M, 3) or (M, 4) surface faces (triangles or quads)
    """
    # First generate the full grid of points
    grid = np.array([
        [i, j, k]
        for i in range(width)
        for j in range(height)
        for k in range(depth)
        if i == 0 or i == width - 1 or j == 0 or j == height - 1 or k == 0 or k == depth - 1
    ], dtype=float)

    # Map points to a dict for fast index lookup
    point_to_index = {tuple(p): idx for idx, p in enumerate(grid)}

    # Optionally: you can rescale to unit cube or center
    V_surface = grid.copy()

    # Construct faces manually for each of the 6 sides of the box
    faces = []

    # Build faces on each face of the box (as quads, converted to triangles)
    def add_face(p0, p1, p2, p3):
        # Add two triangles per quad face
        faces.append([point_to_index[tuple(p0)], point_to_index[tuple(p1)], point_to_index[tuple(p2)]])
        faces.append([point_to_index[tuple(p0)], point_to_index[tuple(p2)], point_to_index[tuple(p3)]])

    # Loop through all surface planes
    for i in range(width - 1):
        for j in range(height - 1):
            # XY plane (Z=0 and Z=depth-1)
            for k in [0, depth - 1]:
                p0 = [i, j, k]
                p1 = [i + 1, j, k]
                p2 = [i + 1, j + 1, k]
                p3 = [i, j + 1, k]
                if all(tuple(p) in point_to_index for p in [p0, p1, p2, p3]):
                    add_face(p0, p1, p2, p3)

    for i in range(width - 1):
        for k in range(depth - 1):
            # XZ plane (Y=0 and Y=height-1)
            for j in [0, height - 1]:
                p0 = [i, j, k]
                p1 = [i + 1, j, k]
                p2 = [i + 1, j, k + 1]
                p3 = [i, j, k + 1]
                if all(tuple(p) in point_to_index for p in [p0, p1, p2, p3]):
                    add_face(p0, p1, p2, p3)

    for j in range(height - 1):
        for k in range(depth - 1):
            # YZ plane (X=0 and X=width-1)
            for i in [0, width - 1]:
                p0 = [i, j, k]
                p1 = [i, j + 1, k]
                p2 = [i, j + 1, k + 1]
                p3 = [i, j, k + 1]
                if all(tuple(p) in point_to_index for p in [p0, p1, p2, p3]):
                    add_face(p0, p1, p2, p3)

    F = np.array(faces, dtype=int)
    return V_surface, F

def tetrahedralize(V, F):
    tetgen_options = "pq1.5Y"  # or "pq1.414a0.01"
    from igl import edges, boundary_facets, barycenter, winding_number, copyleft

    TV, TT, TF = copyleft.tetgen.tetrahedralize(V, F, switches=tetgen_options)

    # if not success:
    #     print("[ERROR] Tetrahedralization failed.")
    #     return

    TT = TT[:, ::-1]  # reverse rows
    TF = TF[:, ::-1]

    BC = barycenter(TV, TT)
    W = winding_number(V, F, BC)

    inside = (W > 0.5)
    IT = TT[inside]  # tets

    faces = boundary_facets(IT)
    F = faces[:, ::-1]

    return TV, IT, F


def get_simple_bar_model_with_surface_points_only(width, height, depth):
    V, F = generate_surface_mesh(width, height, depth)

    V_n, T, F_n = tetrahedralize(V, F)
    return V_n, T, F_n


def volume_of_tet(v0, v1, v2, v3):
    return abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0

def compute_lumped_mass_matrix(V, T, density=1.0):
    from scipy.sparse import coo_matrix

    n = V.shape[0]
    mass_per_vertex = np.zeros(n)

    for tet in T:
        v0, v1, v2, v3 = V[tet]
        vol = volume_of_tet(v0, v1, v2, v3)
        lumped_mass = density * vol / 4.0  # equally divided over 4 verts
        for i in tet:
            mass_per_vertex[i] += lumped_mass
        # Normalize: total mass becomes 1
    total_mass = mass_per_vertex.sum()
    if total_mass > 0:
        mass_per_vertex /= total_mass
    return coo_matrix((mass_per_vertex, (range(n), range(n))), shape=(n, n))

def get_simple_bar_model(width, height, depth):
    """
    Generate a simple 3D bar mesh made of tetrahedra.

    Parameters:
    - width, height, depth: Dimensions of the bar in grid units

    Returns:
    - V: (N, 3) all vertex positions
    - T: (M, 4) tetrahedral elements
    - F: (K, 3) boundary triangle facets (surface)
    - V_surface: (S, 3) surface vertex positions (subset of V)
    """
    # Generate vertex grid
    V = np.zeros((width * height * depth, 3))
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                row = i * height * depth + j * depth + k
                V[row] = [float(i), float(j), float(k)]

    # Build tetrahedra
    tet_count = (width - 1) * (height - 1) * (depth - 1) * 5
    T = np.zeros((tet_count, 4), dtype=int)
    index = 0
    for i in range(width - 1):
        for j in range(height - 1):
            for k in range(depth - 1):
                p0 = i * height * depth + j * depth + k
                p1 = (i + 1) * height * depth + j * depth + k
                p2 = (i + 1) * height * depth + (j + 1) * depth + k
                p3 = i * height * depth + (j + 1) * depth + k
                p4 = i * height * depth + j * depth + (k + 1)
                p5 = (i + 1) * height * depth + j * depth + (k + 1)
                p6 = (i + 1) * height * depth + (j + 1) * depth + (k + 1)
                p7 = i * height * depth + (j + 1) * depth + (k + 1)

                if (i + j + k) % 2 == 1:
                    T[index + 0] = [p1, p0, p5, p2]
                    T[index + 1] = [p5, p2, p7, p6]
                    T[index + 2] = [p7, p0, p5, p4]
                    T[index + 3] = [p2, p0, p7, p3]
                    T[index + 4] = [p5, p0, p7, p2]
                else:
                    T[index + 0] = [p3, p1, p4, p0]
                    T[index + 1] = [p6, p1, p3, p2]
                    T[index + 2] = [p4, p1, p6, p5]
                    T[index + 3] = [p6, p3, p4, p7]
                    T[index + 4] = [p3, p1, p6, p4]
                index += 5

    # Get boundary triangle faces
    F = igl.boundary_facets(T)
    T = T[:, ::-1]  # Reverse winding order
    F = F[:, ::-1]

    # Extract unique vertex indices used in surface triangles
    surface_vertex_indices = np.unique(F.flatten())
    V_surface = V[surface_vertex_indices]

    return V, T, F, V_surface



def get_simple_cloth_model(rows, cols):
    """
    Generate a simple cloth grid mesh in the XY plane.

    Parameters:
    - rows: number of rows
    - cols: number of columns

    Returns:
    - V: (N, 3) vertex positions
    - F: (M, 3) triangle faces
    """
    cloth_positions = []
    cloth_faces = []

    for i in range(rows):
        for j in range(cols):
            xoffset = float(i)
            yoffset = float(j)
            cloth_positions.append([xoffset, yoffset, 0.0])

            if i == rows - 1 or j == cols - 1:
                continue

            ll = i * cols + j
            ul = i * cols + (j + 1)
            lr = (i + 1) * cols + j
            ur = (i + 1) * cols + (j + 1)

            cloth_faces.append([ll, ur, ul])
            cloth_faces.append([ll, lr, ur])

    V = np.array(cloth_positions)
    F = np.array(cloth_faces, dtype=int)
    return V, F