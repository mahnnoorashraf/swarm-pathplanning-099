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
    def __init__(self, grid, start, goal, clearance=0.15, samples_per_seg=64):
        self.grid = grid
        self.n = grid.shape[0]
        self.start = np.array(start, dtype=float) + 0.5
        self.goal = np.array(goal, dtype=float) + 0.5
        self.clearance = clearance       # keeps the path off obstacle corners
        self.M = samples_per_seg         # collision-check samples per segment
        c = clearance
        self._offsets = np.array([[0, 0], [c, c], [c, -c], [-c, c], [-c, -c]])
        self._t = np.linspace(0, 1, self.M)[None, None, :, None]

    # ---------- geometry ----------
    def full_paths(self, X):
        """X: (N, 2K) -> paths (N, K+2, 2)"""
        X = np.atleast_2d(X)
        N = X.shape[0]
        s = np.broadcast_to(self.start, (N, 1, 2))
        g = np.broadcast_to(self.goal, (N, 1, 2))
        return np.concatenate([s, X.reshape(N, -1, 2), g], axis=1)

    def full_path(self, position):
        return self.full_paths(position)[0]

    @staticmethod
    def path_length(path):
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def _segment_info(self, X):
        """Returns per-segment length (N, K+1) and length inside obstacles (N, K+1)."""
        P = self.full_paths(X)
        A, B = P[:, :-1], P[:, 1:]
        L = np.linalg.norm(B - A, axis=2)
        pts = A[:, :, None, :] + self._t * (B - A)[:, :, None, :]   # (N, S, M, 2)
        blocked = np.zeros(pts.shape[:3], dtype=bool)
        for off in self._offsets:
            p = pts + off
            out = (p < 0).any(-1) | (p >= self.n).any(-1)
            ij = np.clip(np.floor(p).astype(int), 0, self.n - 1)
            blocked |= out | self.grid[ij[..., 0], ij[..., 1]]
        inside = blocked.mean(axis=2) * L
        return L, inside

    # ---------- fitness ----------
    def cost(self, X):
        """Vectorised fitness for a whole swarm (or a single particle)."""
        L, inside = self._segment_info(X)
        return L.sum(1) + PENALTY * inside.sum(1) + HIT_COST * (inside > 0).sum(1)

    def collisions(self, position):
        _, inside = self._segment_info(position)
        return int((inside > 0).sum())

    def is_feasible(self, position):
        return self.collisions(position) == 0


class PSO:
    """Global-best PSO with linearly decreasing inertia weight."""

    def __init__(self, problem, n_waypoints=6, n_particles=80, iterations=400,
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
        V = np.zeros_like(X)
        return X, V

    def run(self, verbose=True):
        X, V = self._init_swarm()
        f = self.p.cost(X)
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

            # evaluate (fitness includes the obstacle-collision check)
            f = self.p.cost(X)
            better = f < pbest_f
            pbest[better], pbest_f[better] = X[better], f[better]
            g = int(np.argmin(pbest_f))
            if pbest_f[g] < gbest_f:
                gbest, gbest_f = pbest[g].copy(), pbest_f[g]
            history.append(gbest_f)

            if verbose and (it % 100 == 0 or it == 1):
                print(f"    iter {it:4d} | best cost = {gbest_f:8.3f}")

        return gbest, float(gbest_f), history


def multi_start_pso(problem, runs=5, seed=99, verbose=True, **kw):
    """PSO can get stuck crossing a thin wall (a local minimum). Running a few
    independent swarms and keeping the best collision-free result fixes this."""
    best = None
    for r in range(runs):
        if verbose:
            print(f"  swarm {r + 1}/{runs}")
        pos, f, hist = PSO(problem, seed=seed + r, **kw).run(verbose)
        feasible = problem.is_feasible(pos)
        if verbose:
            print(f"    -> cost {f:.3f}  {'collision-free' if feasible else 'COLLIDES'}")
        key = (not feasible, f)
        if best is None or key < best[0]:
            best = (key, pos, f, hist)
    return best[1], best[2], best[3]
