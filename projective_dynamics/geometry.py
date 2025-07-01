# pd/geometry/get_simple_bar_model.py

import numpy as np
import igl

def get_simple_bar_model(width, height, depth):
    """
    Generate a simple 3D bar mesh made of tetrahedra.

    Parameters:
    - width, height, depth: Dimensions of the bar in grid units

    Returns:
    - V: (N, 3) vertex positions
    - T: (M, 4) tetrahedral elements
    - F: (K, 3) boundary triangle facets
    """
    V = np.zeros((width * height * depth, 3))
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                row = i * height * depth + j * depth + k
                V[row] = [float(i), float(j), float(k)]

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

    F = igl.boundary_facets(T)
    T = T[:, ::-1]  # reverse winding order for consistency
    F = F[:, ::-1]
    return V, T, F



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