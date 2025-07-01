# pd/solver.py

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def flatten(p: np.ndarray) -> np.ndarray:
    """
    Converts an (N, 3) matrix to a (3N,) vector by stacking rows.
    """
    return p.reshape(-1)

def unflatten(q: np.ndarray) -> np.ndarray:
    """
    Converts a (3N,) vector back to an (N, 3) matrix.
    """
    return q.reshape(-1, 3)

class Solver:
    def __init__(self):
        self.model = None
        self.dirty = True
        self.A = None
        self.cholesky = None
        self.dt = None

    def set_model(self, model):
        self.model = model
        self.set_dirty()

    def set_dirty(self):
        self.dirty = True

    def set_clean(self):
        self.dirty = False

    def ready(self):
        return not self.dirty

    def prepare(self, dt):
        self.dt = dt
        positions = self.model.positions
        mass = self.model.mass
        N = positions.shape[0]

        dt2_inv = 1.0 / (dt * dt)
        A_triplets = []

        for constraint in self.model.constraints:
            A_triplets += constraint.get_wi_SiT_AiT_Ai_Si(positions, mass)

        for i in range(N):
            A_triplets.append((3 * i + 0, 3 * i + 0, mass[i] * dt2_inv))
            A_triplets.append((3 * i + 1, 3 * i + 1, mass[i] * dt2_inv))
            A_triplets.append((3 * i + 2, 3 * i + 2, mass[i] * dt2_inv))

        rows, cols, data = zip(*A_triplets)
        A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(3 * N, 3 * N))

        self.cholesky = scipy.sparse.linalg.factorized(A)
        self.set_clean()

    def step(self, fext, num_iterations=10):
        positions = self.model.positions
        velocities = self.model.velocity
        mass = self.model.mass
        constraints = self.model.constraints
        N = positions.shape[0]

        dt = self.dt
        dt_inv = 1.0 / dt
        dt2 = dt * dt
        dt2_inv = 1.0 / dt2

        a = fext / mass[:, None]  # elementwise divide
        explicit = positions + dt * velocities + dt2 * a
        sn = flatten(explicit)

        masses = np.zeros(3 * N)
        for i in range(N):
            masses[3 * i:3 * i + 3] = dt2_inv * mass[i] * sn[3 * i:3 * i + 3]

        q = sn.copy()

        for _ in range(num_iterations):
            b = np.zeros(3 * N)
            for constraint in constraints:
                constraint.project_wi_SiT_AiT_Bi_pi(q, b)
            b += masses

            q = self.cholesky(b)

        q_next = unflatten(q)
        self.model.velocity = (q_next - positions) * dt_inv
        self.model.positions = q_next
