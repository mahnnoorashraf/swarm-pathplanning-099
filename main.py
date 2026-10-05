"""
Swarm Intelligence Lab - Assignment 1
PSO path planning on a 2D grid with obstacles.

Student : Mahnoor
Roll no.: 099   ->   SEED = 99

Run:  python main.py                (uses seed 99)
      python main.py --seed 99 --runs 5 --iters 400
"""

import argparse
import os
import time

import numpy as np

from grid import ROLL_NUMBER, generate_problem
from pso import PathProblem, multi_start_pso
from visualize import plot_convergence, plot_path


def main():
    ap = argparse.ArgumentParser(description="PSO path planning with obstacles")
    ap.add_argument("--seed", type=int, default=ROLL_NUMBER, help="roll number used as seed")
    ap.add_argument("--size", type=int, default=20, help="grid size (size x size)")
    ap.add_argument("--waypoints", type=int, default=6)
    ap.add_argument("--particles", type=int, default=80)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--runs", type=int, default=5, help="independent swarms")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # 1) problem instance from the roll-number seed
    grid, start, goal = generate_problem(seed=args.seed, size=args.size)
    print(f"Seed (roll no.) : {args.seed}")
    print(f"Grid            : {args.size} x {args.size}, {int(grid.sum())} obstacle cells")
    print(f"Start -> Goal   : {start} -> {goal}\n")

    # 2) PSO
    problem = PathProblem(grid, start, goal)
    t0 = time.time()
    best, cost, history = multi_start_pso(
        problem, runs=args.runs, seed=args.seed,
        n_waypoints=args.waypoints, n_particles=args.particles, iterations=args.iters)
    elapsed = time.time() - t0

    path = problem.full_path(best)
    length = problem.path_length(path)
    collisions = problem.collisions(best)

    # 3) report
    print("\n========== RESULT ==========")
    print(f"Best cost        : {cost:.3f}")
    print(f"Path length      : {length:.3f} cells")
    print(f"Collisions       : {collisions} ({'collision-free' if collisions == 0 else 'NOT feasible'})")
    print(f"Straight-line    : {np.linalg.norm(problem.goal - problem.start):.3f} (lower bound)")
    print(f"Time             : {elapsed:.1f} s")
    print("Path (x, y):")
    for p in path:
        print(f"  ({p[0]:6.2f}, {p[1]:6.2f})")

    # 4) plots
    os.makedirs(args.out, exist_ok=True)
    title = f"PSO path — seed {args.seed} — length {length:.2f}"
    plot_path(grid, start, goal, path, title, os.path.join(args.out, "path.png"))
    plot_convergence(history, os.path.join(args.out, "convergence.png"))
    print(f"\nSaved plots to {args.out}/path.png and {args.out}/convergence.png")


if __name__ == "__main__":
    main()
