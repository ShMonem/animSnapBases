import numpy as np
import heapq

STVD_MODE_GRAPH_DIJKSTRA = 0
STVD_MODE_ST_DIJKSTRA = 1
STVD_DEFAULT_K = 10

class STVD:
    def __init__(self):
        self.num_verts = 0
        self.for_tets = False
        self.sources = []
        self.neighbours = []
        self.k = STVD_DEFAULT_K
        self.mode = STVD_MODE_GRAPH_DIJKSTRA
        self.is_updated = False
        self.distances = None
        self.positions = None
        self.outer_vert_normals = None


    def init(self, verts, tris=None, tets=None):
        if tris is None:
            tris = np.empty((0, 3), dtype=int)
        if tets is None:
            tets = np.empty((0, 4), dtype=int)

        self.num_verts = verts.shape[0]
        self.positions = verts
        self.distances = np.full(self.num_verts, -1.0)
        self.is_updated = False
        self.neighbours = [[] for _ in range(self.num_verts)]

        if tets.shape[0] > 0:
            self.for_tets = True
            for tet in tets:
                for i in range(4):
                    for j in range(4):
                        if i != j and tet[j] not in self.neighbours[tet[i]]:
                            self.neighbours[tet[i]].append(tet[j])
        else:
            self.for_tets = False
            self.outer_vert_normals = get_vertex_normals(tris, verts)
            for tri in tris:
                for i in range(3):
                    for j in range(3):
                        if i != j and tri[j] not in self.neighbours[tri[i]]:
                            self.neighbours[tri[i]].append(tri[j])


    def reset_sources(self):
        self.sources.clear()

    def add_source(self, vertex_index):
        if vertex_index < self.num_verts:
            self.is_updated = False
            self.sources.append(vertex_index)

    def reset_distances(self):
        self.distances.fill(-1.0)

    def get_distance(self, v_index):
        if not self.is_updated:
            print("Warning: distances not updated!")
            return -1
        if v_index >= self.num_verts:
            print("Warning: index out of bounds!")
            return -1
        return self.distances[v_index]

    def get_distances(self):
        if not self.is_updated:
            print("Warning: distances not updated!")
        return self.distances

    def compute_distances(self, update=False, mode=STVD_MODE_GRAPH_DIJKSTRA, max_dist=-1.0):
        self.mode = mode
        if not update:
            self.distances.fill(-1.0)

        predecessors = [-1] * self.num_verts
        is_final = [False] * self.num_verts
        queue = []

        for v in self.sources:
            self.distances[v] = 0.0
            heapq.heappush(queue, (0.0, v))

        while queue:
            dist, u = heapq.heappop(queue)
            if is_final[u]:
                continue
            is_final[u] = True

            if max_dist > 0 and self.distances[u] > max_dist:
                continue

            for v in self.neighbours[u]:
                if is_final[v]:
                    continue
                new_dist = self.update_vert_dist(u, v, predecessors)
                if self.distances[v] < 0 or new_dist < self.distances[v]:
                    self.distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(queue, (new_dist, v))

        self.is_updated = True

    def update_vert_dist(self, v1, v2, predecessors):
        p1 = self.positions[v1]
        p2 = self.positions[v2]
        if self.mode == STVD_MODE_GRAPH_DIJKSTRA:
            return self.distances[v1] + np.linalg.norm(p1 - p2)

        elif self.mode == STVD_MODE_ST_DIJKSTRA:
            # Campen unfolding heuristic (only for surface meshes)
            prev_edge = p2 - p1
            e_sum3d = prev_edge.copy()
            best_dist = self.distances[v1] + np.linalg.norm(prev_edge)

            tmp_pred = predecessors[v2]
            predecessors[v2] = v1

            cur_angle = 0.0
            e_sum2d = np.array([0.0, np.linalg.norm(prev_edge)])
            cur_v = v1
            prev_edge_3d = prev_edge

            for i in range(2, self.k + 1):
                pred = predecessors[cur_v]
                if pred < 0:
                    break

                next_edge = self.positions[cur_v] - self.positions[pred]
                if not self.for_tets:
                    n = self.outer_vert_normals[cur_v]
                    next_flat = next_edge - n * np.dot(n, next_edge)
                    prev_flat = prev_edge_3d - n * np.dot(n, prev_edge_3d)
                    angle = get_signed_angle(prev_flat, next_flat, n)
                    cur_angle += angle
                    l = np.linalg.norm(next_flat)
                    next_2d = np.array([np.sin(cur_angle) * l, np.cos(cur_angle) * l])
                    e_sum2d += next_2d
                    cur_dist = self.distances[pred] + np.linalg.norm(e_sum2d)
                else:
                    e_sum3d += next_edge
                    cur_dist = self.distances[pred] + np.linalg.norm(e_sum3d)

                if 0 <= cur_dist < best_dist:
                    best_dist = cur_dist

                prev_edge_3d = next_edge
                cur_v = pred

            return best_dist

        return self.distances[v1] + np.linalg.norm(p1 - p2)

def get_signed_angle(v1, v2, n):
    cross = np.cross(v1, v2)
    sign = np.sign(np.dot(n, cross))
    cos_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return sign * np.arccos(cos_angle)

def get_vertex_normals(triangles, positions, num_outer_vertices=-1):
    if num_outer_vertices <= 0:
        num_outer_vertices = positions.shape[0]

    vert_normals = np.zeros((num_outer_vertices, 3), dtype=np.float64)

    for tri in triangles:
        p0, p1, p2 = positions[tri]
        tn = np.cross(p1 - p0, p2 - p0)
        area = 0.5 * np.linalg.norm(tn)
        if area > 0:
            tn /= np.linalg.norm(tn)  # normalize triangle normal
        for v in tri:
            vert_normals[v] += tn * area

    # Normalize vertex normals
    norms = np.linalg.norm(vert_normals, axis=1)
    nonzero = norms > 1e-8
    vert_normals[nonzero] /= norms[nonzero][:, np.newaxis]

    return vert_normals
