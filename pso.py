"""
Particle Swarm Optimization for 2D path planning.

Encoding
--------
Each particle is a candidate path described by K intermediate waypoints:
    position = [x1, y1, x2, y2, ..., xK, yK]   (continuous, inside the grid)
The full path is  start -> w1 -> w2 -> ... -> wK -> goal.

Cell (i, j) of the grid occupies the square [i, i+1) x [j, j+1); start/goal
sit at cell centres (i+0.5, j+0.5).

Fitness (minimise)
------------------
    cost = path_length + PENALTY * (length of path lying inside obstacles)
                       + HIT_COST * (number of colliding segments)
A collision-free path therefore always beats any colliding one.
"""

import numpy as np

PENALTY = 100.0
HIT_COST = 50.0


class PathProblem:
    def __init__(self, grid, start, goal, clearance=0.15, step=0.05):
        self.grid = grid
        self.n = grid.shape[0]
        self.start = np.array(start, dtype=float) + 0.5
        self.goal = np.array(goal, dtype=float) + 0.5
        self.clearance = clearance  # keeps the path off obstacle corners
        self.step = step            # sampling resolution for collision checks
        c = clearance
        self._offsets = np.array([[0, 0], [c, c], [c, -c], [-c, c], [-c, -c]])

    # ---------- geometry ----------
    def full_path(self, position):
        wps = position.reshape(-1, 2)
        return np.vstack([self.start, wps, self.goal])

    @staticmethod
    def path_length(path):
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def _blocked(self, pts):
        """Boolean per sample point: True if the point (or its clearance box) hits
        an obstacle or leaves the map."""
        hit = np.zeros(len(pts), dtype=bool)
        for off in self._offsets:
            p = pts + off
            out = (p < 0).any(axis=1) | (p >= self.n).any(axis=1)
            ij = np.clip(np.floor(p).astype(int), 0, self.n - 1)
            hit |= out | self.grid[ij[:, 0], ij[:, 1]]
        return hit

    def segment_collision(self, a, b):
        """Length of segment a->b that lies inside obstacles (approx.)."""
        L = np.linalg.norm(b - a)
        k = max(int(L / self.step), 1)
        t = np.linspace(0, 1, k + 1)[:, None]
        pts = a + t * (b - a)
        return self._blocked(pts).sum() * (L / k)

    def collision_info(self, path):
        inside, hits = 0.0, 0
        for a, b in zip(path[:-1], path[1:]):
            c = self.segment_collision(a, b)
            if c > 0:
                inside += c
                hits += 1
        return inside, hits

    def cost(self, position):
        path = self.full_path(position)
        inside, hits = self.collision_info(path)
        return self.path_length(path) + PENALTY * inside + HIT_COST * hits

    def is_feasible(self, position):
        return self.collision_info(self.full_path(position))[1] == 0


class PSO:
    def __init__(self, problem, n_waypoints=5, n_particles=60, iterations=300,
                 w_max=0.9, w_min=0.4, c1=1.6, c2=1.6, v_max_frac=0.15, seed=99):
        self.p = problem
        self.K = n_waypoints
        self.N = n_particles
        self.T = iterations
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.dim = 2 * n_waypoints
        self.lo, self.hi = 0.0, float(problem.n) - 1e-6
        self.v_max = v_max_frac * problem.n
        self.rng = np.random.default_rng(seed)

    def _init_swarm(self):
        """Half the swarm starts near the straight line start->goal (with noise),
        the other half is spread uniformly over the map for exploration."""
        X = np.empty((self.N, self.dim))
        t = np.linspace(0, 1, self.K + 2)[1:-1][:, None]
        line = (self.p.start + t * (self.p.goal - self.p.start)).ravel()
        half = self.N // 2
        X[:half] = line + self.rng.normal(0, 0.15 * self.p.n, (half, self.dim))
        X[half:] = self.rng.uniform(self.lo, self.hi, (self.N - half, self.dim))
        X = np.clip(X, self.lo, self.hi)
        V = self.rng.uniform(-self.v_max, self.v_max, (self.N, self.dim)) * 0.1
        return X, V

    def run(self, verbose=True):
        X, V = self._init_swarm()
        f = np.array([self.p.cost(x) for x in X])
        pbest, pbest_f = X.copy(), f.copy()
        g = int(np.argmin(f))
        gbest, gbest_f = X[g].copy(), f[g]
        history = [gbest_f]

        for it in range(1, self.T + 1):
            w = self.w_max - (self.w_max - self.w_min) * it / self.T  # linear decay
            r1 = self.rng.random((self.N, self.dim))
            r2 = self.rng.random((self.N, self.dim))

            # velocity & position update
            V = w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            V = np.clip(V, -self.v_max, self.v_max)
            X = np.clip(X + V, self.lo, self.hi)

            # evaluate (includes obstacle-collision check)
            f = np.array([self.p.cost(x) for x in X])
            better = f < pbest_f
            pbest[better], pbest_f[better] = X[better], f[better]
            g = int(np.argmin(pbest_f))
            if pbest_f[g] < gbest_f:
                gbest, gbest_f = pbest[g].copy(), pbest_f[g]
            history.append(gbest_f)

            if verbose and (it % 50 == 0 or it == 1):
                print(f"iter {it:4d} | best cost = {gbest_f:8.3f}")

        return gbest, gbest_f, history
