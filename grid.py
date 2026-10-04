"""
Problem generation for PSO path planning.

Everything (grid size, obstacles, start, goal) is generated programmatically
from the roll-number seed, so every student gets a different instance.
"""

import random
from collections import deque

import numpy as np

ROLL_NUMBER = 99  # Roll no. 099  ->  random.seed(99)


def _is_connected(grid, start, goal):
    """BFS (8-connected, no corner cutting) to make sure the goal is reachable."""
    n = grid.shape[0]
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n) or grid[nx, ny]:
                    continue
                # do not squeeze diagonally between two blocked cells
                if dx and dy and (grid[x + dx, y] or grid[x, y + dy]):
                    continue
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
    return False


def generate_problem(seed=ROLL_NUMBER, size=20, density=0.22, n_walls=4):
    """
    Build a grid problem from the given seed.

    Returns
    -------
    grid  : (size, size) bool array, True = obstacle. Indexed grid[x, y].
    start : (x, y) free cell
    goal  : (x, y) free cell
    """
    random.seed(seed)

    while True:  # retries are deterministic because they use the same seeded stream
        grid = np.zeros((size, size), dtype=bool)

        # 1) a few straight wall segments so the problem is not trivial
        for _ in range(n_walls):
            length = random.randint(size // 4, size // 2)
            x0, y0 = random.randrange(size), random.randrange(size)
            if random.random() < 0.5:  # horizontal
                for i in range(length):
                    if x0 + i < size:
                        grid[x0 + i, y0] = True
            else:  # vertical
                for i in range(length):
                    if y0 + i < size:
                        grid[x0, y0 + i] = True

        # 2) scattered single-cell obstacles until the target density is reached
        target = int(density * size * size)
        while grid.sum() < target:
            grid[random.randrange(size), random.randrange(size)] = True

        # 3) start and goal on free cells, reasonably far apart
        free = [(x, y) for x in range(size) for y in range(size) if not grid[x, y]]
        start = random.choice(free)
        far = [c for c in free
               if abs(c[0] - start[0]) + abs(c[1] - start[1]) >= int(1.2 * size)]
        if not far:
            continue
        goal = random.choice(far)

        if _is_connected(grid, start, goal):
            return grid, start, goal


if __name__ == "__main__":
    g, s, t = generate_problem()
    print(f"seed={ROLL_NUMBER}  grid={g.shape}  obstacles={g.sum()}  start={s}  goal={t}")
