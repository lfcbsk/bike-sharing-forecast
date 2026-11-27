"""Create the three compact explanatory diagrams used in the Task 2 report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent.parent / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, width, height, label, color="#E8EEF4", fontsize=11):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02",
                           linewidth=1, edgecolor="#50657A", facecolor=color)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.3, color="#50657A"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 2.3))
ax.set(xlim=(0, 10), ylim=(0, 2.3))
ax.axis("off")
box(ax, 0.3, 0.7, 4.0, 0.9, "Observed hours  t−23 … t\n24 hours × 19 features")
box(ax, 5.7, 0.7, 4.0, 0.9, "Forecast hours  t+1 … t+5\n5 rental counts", "#E9F3E9")
arrow(ax, 4.45, 1.15, 5.55, 1.15)
save(fig, "problem_formulation.png")

fig, ax = plt.subplots(figsize=(11, 2.7))
ax.set(xlim=(0, 11), ylim=(0, 2.7))
ax.axis("off")
items = [
    (0.15, "24 × 19\nhistory"), (2.35, "Two-layer\nLSTM encoder"),
    (4.55, "24 encoder\nstates"), (6.75, "Bahdanau\nattention"),
    (8.95, "LSTM decoder\n5 forecasts"),
]
for x, label in items:
    box(ax, x, 0.8, 1.85, 1.0, label, fontsize=10)
for x in (2.0, 4.2, 6.4, 8.6):
    arrow(ax, x + 0.02, 1.3, x + 0.3, 1.3)
ax.text(9.9, 0.35, "Starts with last observed count; then uses its own predictions",
        ha="center", fontsize=9)
save(fig, "architecture_overview.png")

fig, ax = plt.subplots(figsize=(11, 2.7))
ax.set(xlim=(0, 11), ylim=(0, 2.7))
ax.axis("off")
labels = ["Sort and\nengineer", "Split and\nscale", "24-to-5\nwindows",
          "Train and\nvalidate", "Best checkpoint\nand test metrics"]
for index, label in enumerate(labels):
    x = 0.15 + 2.2 * index
    box(ax, x, 0.8, 1.85, 1.0, label, fontsize=10)
    if index < len(labels) - 1:
        arrow(ax, x + 1.9, 1.3, x + 2.15, 1.3)
save(fig, "experimental_workflow.png")

print(OUT)
