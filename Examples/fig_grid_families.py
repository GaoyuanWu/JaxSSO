"""
Generate diagrams showing the bar families in quad and tri grids,
each family in a different color.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# =========================================================================
# 1. Regular quad grid with 2 bar families
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.set_title("Quad grid: 2 bar families", fontsize=14)

nx, ny = 6, 5
# Draw horizontal bars (family 1, 0 deg)
for j in range(ny):
    for i in range(nx - 1):
        ax.plot([i, i + 1], [j, j], color="C3", linewidth=2.5, solid_capstyle="round")

# Draw vertical bars (family 2, 90 deg)
for j in range(ny - 1):
    for i in range(nx):
        ax.plot([i, i], [j, j + 1], color="C0", linewidth=2.5, solid_capstyle="round")

# Nodes
for j in range(ny):
    for i in range(nx):
        ax.plot(i, j, "ko", markersize=4, zorder=5)

# Spacing annotations
ax.annotate(
    "", xy=(1, -0.6), xytext=(0, -0.6),
    arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2),
)
ax.text(0.5, -0.85, "$d$", fontsize=14, color="gray", ha="center")

# Legend
ax.plot([], [], color="C3", linewidth=2.5, label=r"Family 1: $\theta = 0°$")
ax.plot([], [], color="C0", linewidth=2.5, label=r"Family 2: $\theta = 90°$")
ax.legend(fontsize=11, loc="upper left", framealpha=0.9)

ax.set_xlim(-0.8, nx - 0.2)
ax.set_ylim(-1.3, ny - 0.2)
ax.set_aspect("equal")
ax.axis("off")

# =========================================================================
# 2. Regular triangular grid with 3 bar families
# =========================================================================
ax = axes[1]
ax.set_title("Tri grid: 3 bar families", fontsize=14)

nx, ny = 6, 5
# Generate triangle grid points
pts = []
for j in range(ny):
    for i in range(nx):
        x = i + 0.5 * (j % 2)
        y = j * np.sqrt(3) / 2
        pts.append((x, y))
pts = np.array(pts)

# Build edges and classify into 3 families by angle
edges_by_family = {0: [], 1: [], 2: []}  # 0~0deg, 1~60deg, 2~120deg
for a in range(len(pts)):
    for b in range(a + 1, len(pts)):
        dx = pts[b, 0] - pts[a, 0]
        dy = pts[b, 1] - pts[a, 1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 1.2:
            continue  # skip non-neighbors
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if angle < 15:
            edges_by_family[0].append((a, b))
        elif angle < 45:
            edges_by_family[1].append((a, b))
        else:
            edges_by_family[2].append((a, b))

colors = {"C3": 0, "C0": 1, "C2": 2}
family_colors = ["C3", "C2", "C0"]
family_labels = [
    r"Family 1: $\theta \approx 0°$",
    r"Family 2: $\theta \approx 60°$",
    r"Family 3: $\theta \approx 120°$",
]
# Note: arctan2 gives absolute angle, so 60 and 120 both show as ~60 in abs.
# We need to distinguish by sign of dx.
# Redo classification more carefully:
edges_by_family = {0: [], 1: [], 2: []}
for a in range(len(pts)):
    for b in range(a + 1, len(pts)):
        dx = pts[b, 0] - pts[a, 0]
        dy = pts[b, 1] - pts[a, 1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 1.2:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalize to [0, 180)
        if angle < 0:
            angle += 180
        if angle > 160 or angle < 20:
            edges_by_family[0].append((a, b))  # ~0 deg
        elif 40 < angle < 80:
            edges_by_family[1].append((a, b))  # ~60 deg
        elif 100 < angle < 140:
            edges_by_family[2].append((a, b))  # ~120 deg

for fam_idx in range(3):
    for a, b in edges_by_family[fam_idx]:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            color=family_colors[fam_idx],
            linewidth=2.5,
            solid_capstyle="round",
        )

# Nodes
ax.plot(pts[:, 0], pts[:, 1], "ko", markersize=4, zorder=5)

# Spacing annotation
# d for the horizontal family = vertical distance between rows = sqrt(3)/2
y0 = 0
y1 = np.sqrt(3) / 2
ax.annotate(
    "", xy=(-0.6, y1), xytext=(-0.6, y0),
    arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2),
)
ax.text(-0.85, (y0 + y1) / 2, "$d$", fontsize=14, color="gray", ha="center", va="center")

# Legend
for fam_idx in range(3):
    ax.plot([], [], color=family_colors[fam_idx], linewidth=2.5, label=family_labels[fam_idx])
ax.legend(fontsize=11, loc="upper left", framealpha=0.9)

ax.set_xlim(-1.3, nx + 0.3)
ax.set_ylim(-0.8, (ny - 1) * np.sqrt(3) / 2 + 0.5)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout()
fig.savefig("Examples/fig_grid_families.png", dpi=150, bbox_inches="tight")
print("Saved Examples/fig_grid_families.png")
