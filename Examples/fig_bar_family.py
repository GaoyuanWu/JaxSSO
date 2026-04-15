"""Generate a diagram of a single bar family for NOTES.md."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(7, 5))

theta_deg = 35
theta = np.radians(theta_deg)
n = np.array([np.cos(theta), np.sin(theta)])
perp = np.array([-np.sin(theta), np.cos(theta)])

# Draw parallel bars
n_bars = 5
d = 1.0  # spacing
bar_length = 5.0
center = np.array([3.0, 2.5])

for i in range(n_bars):
    offset = (i - n_bars // 2) * d
    p0 = center + offset * perp - (bar_length / 2) * n
    p1 = center + offset * perp + (bar_length / 2) * n
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", linewidth=2.5, solid_capstyle="round")

# Label d (spacing between two middle bars)
i1, i2 = n_bars // 2, n_bars // 2 + 1
mid1 = center + (i1 - n_bars // 2) * d * perp
mid2 = center + (i2 - n_bars // 2) * d * perp
mid_d = (mid1 + mid2) / 2

# Draw the perpendicular spacing arrow
arr_start = mid1 + 0.15 * perp
arr_end = mid2 - 0.15 * perp
ax.annotate(
    "",
    xy=arr_end,
    xytext=arr_start,
    arrowprops=dict(arrowstyle="<->", color="C0", lw=1.5),
)
ax.text(
    mid_d[0] + 0.25 * perp[0] + 0.1 * n[0],
    mid_d[1] + 0.25 * perp[1] + 0.1 * n[1],
    "$d$",
    fontsize=16,
    color="C0",
    ha="center",
    va="center",
)

# Draw n vector on a middle bar
bar_mid = center
ax.annotate(
    "",
    xy=bar_mid + 1.3 * n,
    xytext=bar_mid,
    arrowprops=dict(arrowstyle="->, head_width=0.3", color="C3", lw=2),
)
ax.text(
    bar_mid[0] + 1.5 * n[0] + 0.15,
    bar_mid[1] + 1.5 * n[1] + 0.1,
    r"$\mathbf{n}$",
    fontsize=16,
    color="C3",
    ha="center",
    va="center",
)

# Draw theta arc
arc_r = 1.0
arc_angles = np.linspace(0, theta, 30)
arc_x = center[0] - bar_length / 2 * n[0] + arc_r * np.cos(arc_angles)
arc_y = center[1] - bar_length / 2 * n[1] + arc_r * np.sin(arc_angles)
ax.plot(arc_x, arc_y, "C2-", linewidth=1.5)
arc_label_angle = theta / 2
ax.text(
    center[0] - bar_length / 2 * n[0] + (arc_r + 0.3) * np.cos(arc_label_angle),
    center[1] - bar_length / 2 * n[1] + (arc_r + 0.3) * np.sin(arc_label_angle),
    r"$\theta$",
    fontsize=15,
    color="C2",
    ha="center",
    va="center",
)

# Draw x-axis reference at the same origin
origin = center - (bar_length / 2) * n
ax.annotate(
    "",
    xy=origin + np.array([1.5, 0]),
    xytext=origin,
    arrowprops=dict(arrowstyle="->, head_width=0.2", color="gray", lw=1.2),
)
ax.text(origin[0] + 1.7, origin[1] - 0.15, "$x$", fontsize=14, color="gray")

# Label: cross-section A, modulus E_b
ax.text(
    0.3,
    4.6,
    r"Each bar: modulus $E_b$, area $A$",
    fontsize=13,
    fontstyle="italic",
    color="k",
)

# Shade a representative strip between two bars to show tributary width
i_strip = n_bars // 2
strip_center = center + (i_strip - n_bars // 2) * d * perp
corners = np.array(
    [
        strip_center - d / 2 * perp - bar_length / 2 * n,
        strip_center - d / 2 * perp + bar_length / 2 * n,
        strip_center + d / 2 * perp + bar_length / 2 * n,
        strip_center + d / 2 * perp - bar_length / 2 * n,
    ]
)
strip = patches.Polygon(corners, closed=True, alpha=0.08, color="C0", linewidth=0)
ax.add_patch(strip)

ax.set_xlim(-1.5, 7.5)
ax.set_ylim(-1, 6)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("A family of parallel bars at angle $\\theta$, spacing $d$", fontsize=14)

fig.tight_layout()
fig.savefig("Examples/fig_bar_family.png", dpi=150, bbox_inches="tight")
print("Saved Examples/fig_bar_family.png")
