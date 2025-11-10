import numpy as np
import igl
import polyscope as ps


# ------------------------------
# Step 1. Load mesh
# ------------------------------
V, F = igl.read_triangle_mesh("../data/sphere.obj")
n = V.shape[0]
print(f"Loaded mesh: {n} vertices, {F.shape[0]} faces")


# ------------------------------
# Step 2. Farthest-point sampling on the surface
# ------------------------------
def farthest_point_sampling(V, F, k):
    """Return k approximately evenly distributed vertex indices using graph-geodesic FPS."""
    n = V.shape[0]

    # Build adjacency graph from triangles
    adj = [[] for _ in range(n)]
    for tri in F:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            adj[a].append(b)
            adj[b].append(a)

    # Precompute edge lengths
    edge_lengths = {}
    for i in range(n):
        for j in adj[i]:
            edge_lengths[(i, j)] = np.linalg.norm(V[i] - V[j])

    def dijkstra(src):
        """Compute shortest path (graph geodesic) distance from src."""
        import heapq
        dist = np.full(n, np.inf)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v in adj[u]:
                alt = d + edge_lengths[(u, v)]
                if alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))
        return dist

    # Farthest point sampling
    seeds = [np.random.randint(0, n)]
    D = dijkstra(seeds[0])
    for _ in range(1, k):
        next_seed = np.argmax(D)
        seeds.append(next_seed)
        D = np.minimum(D, dijkstra(next_seed))

    return np.array(seeds)




k = 10  # number of handles
seeds = farthest_point_sampling(V, F, k)
print("Selected seeds:", seeds)


# ------------------------------
# Step 3. Compute Bounded Biharmonic Weights (BBW)
# ------------------------------
# Harmonic (smooth) weights fallback if BBWData is unavailable
b = seeds
bc = np.eye(len(b))

print("Computing harmonic weights (BBW fallback)...")
W = igl.harmonic(V, F, b, bc, 1)  # 2 = biharmonic, 1 = Laplacian
W = np.maximum(W, 0)              # Clamp to positive
W /= W.sum(axis=1, keepdims=True)
print("Harmonic weights shape:", W.shape)


# ------------------------------
# Step 4. Visualize with Polyscope
# ------------------------------
ps.init()
ps_mesh = ps.register_surface_mesh("mesh", V, F)

# Register weights as scalar fields
for i in range(W.shape[1]):
    ps_mesh.add_scalar_quantity(f"weight_{i}", W[:, i], defined_on='vertices', enabled=(i == 0))

# Register sampled handles
ps.register_point_cloud("handles", V[b], radius=0.01, color=(1, 0, 0))
ps.show()
