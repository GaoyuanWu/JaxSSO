"""
Color the actual Mannheim mesh edges by approximate bar family,
showing how the idealized parallel families map to an irregular mesh.
"""
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# Load mesh
mesh = pv.read("Examples/Data/mannheim_quad.vtu")
pts = np.array(mesh.points)
xs = pts[:, 0] - pts[:, 0].min()
ys = pts[:, 1] - pts[:, 1].min()
n_ele = mesh.n_cells
cnct_quad = mesh.cells.reshape(-1, 5)[:, 1:]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- Quad grid ----
ax = axes[0]
ax.set_title("Mannheim quad grid: 2 edge families by angle", fontsize=13)

quad_edge_set = set()
for q in cnct_quad:
    for a, b in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]:
        quad_edge_set.add((min(a, b), max(a, b)))

for a, b in quad_edge_set:
    dx = xs[b] - xs[a]
    dy = ys[b] - ys[a]
    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
    color = "C3" if angle < 45 else "C0"
    ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=color, linewidth=0.8, alpha=0.8)

ax.plot([], [], color="C3", linewidth=2, label=r"Family 1: mostly horizontal ($\theta < 45°$)")
ax.plot([], [], color="C0", linewidth=2, label=r"Family 2: mostly vertical ($\theta \geq 45°$)")
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax.set_aspect("equal")
ax.axis("off")

# ---- Tri grid ----
ax = axes[1]
ax.set_title("Mannheim tri grid: 3 edge families by angle", fontsize=13)

# Triangulate
quad_pv = pv.UnstructuredGrid(
    np.hstack([np.full((n_ele, 1), 4, dtype=int), cnct_quad]),
    np.full(n_ele, pv.CellType.QUAD),
    np.column_stack([xs, ys, np.zeros(len(xs))]),
)
tri_pv = quad_pv.triangulate()
cnct_tri = tri_pv.cells.reshape(-1, 4)[:, 1:]

tri_edge_set = set()
for tri in cnct_tri:
    for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
        tri_edge_set.add((min(a, b), max(a, b)))

# Classify into 3 families based on angle
# For a quad mesh triangulated by adding one diagonal, we get:
# - original horizontal-ish edges (~0 deg)
# - original vertical-ish edges (~90 deg)
# - new diagonal edges (~45 deg or ~135 deg)
for a, b in tri_edge_set:
    dx = xs[b] - xs[a]
    dy = ys[b] - ys[a]
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 180

    if angle < 30 or angle >= 150:
        color = "C3"  # ~horizontal
    elif 30 <= angle < 75:
        color = "C2"  # ~diagonal up-right
    elif 75 <= angle < 110:
        color = "C0"  # ~vertical
    else:
        color = "C4"  # ~diagonal up-left
    ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=color, linewidth=0.6, alpha=0.8)

ax.plot([], [], color="C3", linewidth=2, label=r"~horizontal ($\theta < 30°$ or $\geq 150°$)")
ax.plot([], [], color="C2", linewidth=2, label=r"~diagonal ($30° \leq \theta < 75°$)")
ax.plot([], [], color="C0", linewidth=2, label=r"~vertical ($75° \leq \theta < 110°$)")
ax.plot([], [], color="C4", linewidth=2, label=r"~diagonal ($110° \leq \theta < 150°$)")
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.set_aspect("equal")
ax.axis("off")

fig.tight_layout()
fig.savefig("Examples/fig_mannheim_families.png", dpi=150, bbox_inches="tight")
print("Saved Examples/fig_mannheim_families.png")
