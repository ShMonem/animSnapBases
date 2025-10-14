import numpy as np
import heapq

STVD_MODE_GRAPH_DIJKSTRA = 0
STVD_MODE_ST_DIJKSTRA = 1
STVD_DEFAULT_K = 4

class STVD:
    def __init__(self):
        self.num_verts = 0
        self.neighbours = []
        self.mode = STVD_MODE_GRAPH_DIJKSTRA
        self.k = STVD_DEFAULT_K
        self.distances = None
        self.positions = None
        self.sources = []
        self.for_tets = False
        self.is_updated = False
        self.outer_vert_normals = None  # only used for triangle meshes

    # ---------------------------------------------------------
    # Initialization: build neighborhood connectivity
    # ---------------------------------------------------------
    def init(self, verts, tris=None, tets=None):
        self.num_verts = verts.shape[0]
        self.positions = verts
        self.neighbours = [[] for _ in range(self.num_verts)]
        self.distances = np.full(self.num_verts, -1.0)
        self.is_updated = False

        if tets is not None and len(tets) > 0:
            self.for_tets = True
            for tet in tets:
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            v1, v2 = int(tet[i]), int(tet[j])
                            if v2 not in self.neighbours[v1]:
                                self.neighbours[v1].append(v2)
        elif tris is not None and len(tris) > 0:
            self.for_tets = False
            for tri in tris:
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            v1, v2 = int(tri[i]), int(tri[j])
                            if v2 not in self.neighbours[v1]:
                                self.neighbours[v1].append(v2)

    # ---------------------------------------------------------
    def reset_sources(self):
        self.sources = []

    def add_source(self, vertex_idx):
        if vertex_idx < self.num_verts:
            self.sources.append(vertex_idx)
            self.is_updated = False

    def reset_distances(self):
        self.distances[:] = -1.0
        self.is_updated = False

    # ---------------------------------------------------------
    def compute_distances(self, update=False, mode=STVD_MODE_GRAPH_DIJKSTRA, max_dist=-1.0):
        self.mode = mode
        if not update:
            self.distances[:] = -1.0

        n = self.num_verts
        self.distances[:] = np.where(self.distances < 0, np.inf, self.distances)

        predecessors = np.full(n, -1, dtype=int)
        is_final = np.zeros(n, dtype=bool)
        pq = []

        for s in self.sources:
            self.distances[s] = 0.0
            heapq.heappush(pq, (0.0, s))

        while pq:
            cur_dist, cur_v = heapq.heappop(pq)
            if is_final[cur_v]:
                continue
            is_final[cur_v] = True

            if max_dist > 0 and cur_dist > max_dist:
                continue

            for nb in self.neighbours[cur_v]:
                if is_final[nb]:
                    continue

                updated_dist = self.update_vert_dist(cur_v, nb, predecessors)
                if np.isnan(updated_dist):
                    continue

                if self.distances[nb] < 0 or updated_dist < self.distances[nb]:
                    self.distances[nb] = updated_dist
                    predecessors[nb] = cur_v
                    heapq.heappush(pq, (updated_dist, nb))

        self.is_updated = True

    # ---------------------------------------------------------
    def update_vert_dist(self, v1, v2, predecessors):
        p1, p2 = self.positions[v1], self.positions[v2]
        base_dist = np.linalg.norm(p1 - p2)
        if self.mode == STVD_MODE_GRAPH_DIJKSTRA or self.for_tets:
            return self.distances[v1] + base_dist

        # ST-Dijkstra unfolding (surface only)
        prev_edge = p2 - p1
        tmp_pred = predecessors[v2]
        predecessors[v2] = v1

        prev_v = v2
        cur_v = v1
        e_sum3d = prev_edge.copy()
        e_len = np.linalg.norm(e_sum3d)
        e_sum2d = np.array([0.0, e_len])
        cur_angle = 0.0
        best_dist = self.distances[v1] + e_len

        for _ in range(2, self.k + 1):
            if predecessors[cur_v] < 0:
                break
            next_v = predecessors[cur_v]
            next_edge = self.positions[cur_v] - self.positions[next_v]

            cur_dist = -1
            n = self.outer_vert_normals[cur_v]
            next_edge_flat = next_edge - n * np.dot(n, next_edge)
            prev_edge_flat = prev_edge - n * np.dot(n, prev_edge)
            angle = self.get_signed_angle(prev_edge_flat, next_edge_flat, n)
            cur_angle += angle
            l = np.linalg.norm(next_edge_flat)
            next_edge_2d = np.array([np.sin(cur_angle) * l, np.cos(cur_angle) * l])
            e_sum2d += next_edge_2d
            cur_dist = self.distances[next_v] + np.linalg.norm(e_sum2d)

            if 0 <= cur_dist < best_dist:
                best_dist = cur_dist

            prev_v = cur_v
            cur_v = next_v
            prev_edge = next_edge

        return best_dist

    @staticmethod
    def get_signed_angle(v1, v2, n):
        v1n = v1 / np.linalg.norm(v1)
        v2n = v2 / np.linalg.norm(v2)
        cross = np.cross(v1n, v2n)
        sign = np.sign(np.dot(cross, n))
        return np.arccos(np.clip(np.dot(v1n, v2n), -1.0, 1.0)) * sign

    def get_distance(self, v_idx):
        if not self.is_updated:
            print("Warning: distances are not up to date.")
        return self.distances[v_idx]

    def get_distances(self):
        return self.distances.copy()
