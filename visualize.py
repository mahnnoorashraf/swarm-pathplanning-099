"""Plotting helpers: grid + obstacles + start/goal + PSO path, and convergence curve."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_path(grid, start, goal, path, title, out_file):
    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7))

    for x in range(n):
        for y in range(n):
            if grid[x, y]:
                ax.add_patch(Rectangle((x, y), 1, 1, color="#3b3b3b"))

    ax.plot(path[:, 0], path[:, 1], "-", color="#1f77b4", lw=2.5, label="PSO path")
    ax.plot(path[1:-1, 0], path[1:-1, 1], "o", color="#ff7f0e", ms=6, label="waypoints")
    ax.plot(start[0] + 0.5, start[1] + 0.5, "s", color="#2ca02c", ms=13, label=f"start {start}")
    ax.plot(goal[0] + 0.5, goal[1] + 0.5, "*", color="#d62728", ms=18, label=f"goal {goal}")

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(0, n + 1, 2))
    ax.set_yticks(range(0, n + 1, 2))
    ax.set_xticks(range(n + 1), minor=True)
    ax.set_yticks(range(n + 1), minor=True)
    ax.grid(which="minor", color="#dddddd", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_convergence(history, out_file):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="#1f77b4", lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("global best cost (log scale)")
    ax.set_title("PSO convergence (best swarm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
