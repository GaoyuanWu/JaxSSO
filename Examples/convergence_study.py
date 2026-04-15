"""
Convergence study: quad shell (MITC4) and tri shell (MITC3) on the Mannheim
Multihalle mesh at 3 refinement levels (0 = original, 1 = subdivide once,
2 = subdivide twice).

Outputs:
  - Strain energy convergence plot
  - Optimized shape side-by-side plots
  - VTU files for each level/model
"""

import numpy as np
import jax
import jax.numpy as jnp
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path

jax.config.update("jax_enable_x64", True)

import JaxSSO.model as Model
import JaxSSO.optimization as optimization
from JaxSSO.SSO_model import NodeParameter, SSO_model

OUT = Path("Examples/output/convergence")
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================================
# 1. Load base mesh
# =========================================================================
mesh = pv.read("Examples/Data/mannheim_quad.vtu")
raw_pts = np.array(mesh.points)
bc_flag_orig = np.array(mesh.point_data["bc_node"])
n_orig = mesh.n_points
n_ele_quad_orig = mesh.n_cells
cnct_quad_orig = mesh.cells.reshape(-1, 5)[:, 1:]

xs_orig = raw_pts[:, 0] - raw_pts[:, 0].min()
ys_orig = raw_pts[:, 1] - raw_pts[:, 1].min()

# Triangulate base mesh: split each quad along its SHORTER diagonal
# to produce better-conditioned triangles (avoids pyvista's arbitrary choice)
def triangulate_short_diagonal(cnct_quad, xs, ys):
    """Split each quad into 2 triangles along the shorter diagonal."""
    tris = []
    for q in cnct_quad:
        n0, n1, n2, n3 = q
        # Diagonal 0-2 vs diagonal 1-3
        d02 = np.sqrt((xs[n2] - xs[n0])**2 + (ys[n2] - ys[n0])**2)
        d13 = np.sqrt((xs[n3] - xs[n1])**2 + (ys[n3] - ys[n1])**2)
        if d02 <= d13:
            tris.append([n0, n1, n2])
            tris.append([n0, n2, n3])
        else:
            tris.append([n0, n1, n3])
            tris.append([n1, n2, n3])
    return np.array(tris, dtype=int)

cnct_tri_orig = triangulate_short_diagonal(cnct_quad_orig, xs_orig, ys_orig)

# Build PolyData for subdivision
tri_pd = pv.PolyData(
    np.column_stack([xs_orig, ys_orig, np.zeros(n_orig)]),
    faces=np.hstack([np.full((len(cnct_tri_orig), 1), 3, dtype=int), cnct_tri_orig]).ravel(),
)

def subdivide_quads(cnct_quad, xs, ys):
    """
    Subdivide each quad into 4 quads by adding edge midpoints and face centers.
    Returns (xs_new, ys_new, cnct_new, bc_new) with new nodes appended.
    """
    n_node = len(xs)
    xs_new = list(xs)
    ys_new = list(ys)

    # Build unique edges and assign midpoint node IDs
    edge_to_mid = {}
    edge_list = []
    for q in cnct_quad:
        for a, b in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]:
            edge = (min(a, b), max(a, b))
            if edge not in edge_to_mid:
                mid_id = n_node + len(edge_list)
                edge_to_mid[edge] = mid_id
                edge_list.append(edge)
                xs_new.append(0.5 * (xs[a] + xs[b]))
                ys_new.append(0.5 * (ys[a] + ys[b]))

    # Face centers
    n_after_edges = len(xs_new)
    cnct_new = []
    for i, q in enumerate(cnct_quad):
        n0, n1, n2, n3 = q
        # Face center
        fc_id = n_after_edges + i
        xs_new.append(0.25 * (xs[n0] + xs[n1] + xs[n2] + xs[n3]))
        ys_new.append(0.25 * (ys[n0] + ys[n1] + ys[n2] + ys[n3]))

        # Edge midpoints
        m01 = edge_to_mid[(min(n0, n1), max(n0, n1))]
        m12 = edge_to_mid[(min(n1, n2), max(n1, n2))]
        m23 = edge_to_mid[(min(n2, n3), max(n2, n3))]
        m30 = edge_to_mid[(min(n3, n0), max(n3, n0))]

        # 4 child quads (counter-clockwise winding preserved)
        cnct_new.append([n0, m01, fc_id, m30])
        cnct_new.append([m01, n1, m12, fc_id])
        cnct_new.append([fc_id, m12, n2, m23])
        cnct_new.append([m30, fc_id, m23, n3])

    return np.array(xs_new), np.array(ys_new), np.array(cnct_new, dtype=int), edge_to_mid


print(f"Base mesh: {n_orig} nodes, {n_ele_quad_orig} quads, {len(cnct_tri_orig)} tris")

# =========================================================================
# 2. Material and optimization parameters
# =========================================================================
t = 0.1
E = 1e10
nu = 0.3
load = 5000.0
N_ITER = {0: 400, 1: 600}  # more iterations for refined meshes
STEP_SIZE = 5e-2  # smaller step to avoid oscillation on coarse meshes


# =========================================================================
# 3. Mesh refinement: generate levels 0, 1, 2
# =========================================================================
def build_level(subdiv_level):
    """
    Build mesh data for a given subdivision level.

    Returns (xs, ys, bc_flag, cnct_tri, cnct_quad_or_None, n_node, label)
    """
    if subdiv_level == 0:
        return (
            xs_orig.copy(), ys_orig.copy(),
            bc_flag_orig.copy(),
            cnct_tri_orig.copy(), cnct_quad_orig.copy(),
        )

    # Subdivide triangulated surface mesh
    refined = tri_pd.subdivide(subdiv_level, subfilter="linear")
    pts_ref = np.array(refined.points)
    xs_r = pts_ref[:, 0] - pts_ref[:, 0].min()
    ys_r = pts_ref[:, 1] - pts_ref[:, 1].min()
    cnct_tri_r = refined.regular_faces
    n_node_r = refined.n_points

    # Identify BC nodes using the geometric boundary of the refined mesh.
    # All nodes on the mesh boundary (edges shared by exactly one cell) are BC.
    boundary = refined.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False
    )
    # Match boundary points back to refined mesh point indices
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_ref)
    dists, idxs = tree.query(np.array(boundary.points))
    boundary_node_ids = set(idxs[dists < 1e-10])

    bc_flag_r = np.zeros(n_node_r, dtype=int)
    for nd in boundary_node_ids:
        bc_flag_r[nd] = 1

    # Also build refined quad mesh by subdividing quads
    cnct_quad_r = None
    if subdiv_level == 1:
        xs_q, ys_q, cnct_quad_r, _ = subdivide_quads(cnct_quad_orig, xs_orig, ys_orig)
        # xs_q/ys_q should match xs_r/ys_r for shared nodes (originals + edge midpoints)
        # but quad subdivision also adds face centers. We need a unified point set.
        # Actually, the quad-refined mesh has DIFFERENT points than the tri-refined mesh
        # (quad adds face centers, tri doesn't). So we return the quad data separately.

    return xs_r, ys_r, bc_flag_r, cnct_tri_r, cnct_quad_r


def extract_edges(cnct_tri):
    """Extract unique edges from triangle connectivity."""
    edge_set = set()
    for f in cnct_tri:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            edge_set.add((min(a, b), max(a, b)))
    return np.array(sorted(edge_set), dtype=int)


def beam_props_tri_homog(E, t, d):
    """Inverse homogenization for a tri grid (3 bar families at 0/60/120 deg)."""
    s3h = np.sqrt(3) / 2
    A = s3h * t * d
    Iy = s3h * t**3 * d / 12
    Iz = Iy
    J = Iy + Iz
    G_b = E / (2 * (1 + 1 / 3))  # nu_eff = 1/3
    return A, Iy, Iz, J, G_b


# =========================================================================
# 4. Filter builder
# =========================================================================
def make_filter(xs, ys, R=10.0):
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D = np.sqrt(dx**2 + dy**2)
    B_raw = np.where(D > R, 0.0, (1.0 / R) * (R - D))
    return B_raw / B_raw.sum(axis=1, keepdims=True)


# =========================================================================
# 5. Shape optimization driver
# =========================================================================
def run_opt(label, xs, ys, bc_flag, cnct, elem_type, B_ij, n_iter=400):
    n_node = len(xs)
    design_nodes = np.where(bc_flag == 0)[0]
    n_free = len(design_nodes)

    # Keep total load constant across refinement levels:
    # total_load = n_free_L0 * load = const, so load_per_node = n_free_L0 * load / n_free
    n_free_L0 = int((bc_flag_orig == 0).sum())
    load_per_node = load * n_free_L0 / n_free

    rng = np.random.RandomState(42)
    zs = np.zeros(n_node)
    zs[design_nodes] = rng.uniform(0.5, 0.51, size=len(design_nodes))
    zs_filt = zs.copy()
    zs_filt[design_nodes] = (B_ij @ zs_filt)[design_nodes]

    model = Model.Model()
    for i in range(n_node):
        model.add_node(i, xs[i], ys[i], zs_filt[i])
        if bc_flag[i]:
            model.add_support(i, [1, 1, 1, 1, 1, 1])
        else:
            model.add_nodal_load(i, nodal_load=[0, 0, -load_per_node, 0, 0, 0])

    if elem_type == "quad":
        for i in range(len(cnct)):
            model.add_quad(i, *cnct[i], t, E, nu)
    elif elem_type == "tri":
        for i in range(len(cnct)):
            model.add_tri(i, cnct[i, 0], cnct[i, 1], cnct[i, 2], t, E, nu)
    elif elem_type == "beam":
        # cnct is edge array (n_edges, 2); use homogenized properties
        pts_flat = np.column_stack([xs, ys, np.zeros(n_node)])
        d_avg = np.linalg.norm(pts_flat[cnct[:, 1]] - pts_flat[cnct[:, 0]], axis=1).mean()
        A_b, Iy_b, Iz_b, J_b, G_b = beam_props_tri_homog(E, t, d_avg)
        for i in range(len(cnct)):
            model.add_beamcol(i, cnct[i, 0], cnct[i, 1], E, G_b, Iy_b, Iz_b, J_b, A_b)

    model.model_ready()
    sso = SSO_model(model)
    for nd in design_nodes:
        sso.add_nodeparameter(NodeParameter(nd, 2))
    sso.initialize_parameters_values()
    sso.set_objective(objective="strain energy")

    C_hist = []

    def objective(x, sso=sso, zs=zs):
        p = zs.copy()
        p[design_nodes] = x
        z = (B_ij @ p)[design_nodes]
        sso.update_nodeparameter(z)
        C, sens = sso.value_grad_params(which_solver="sparse", enforce_scipy_sparse=True)
        C_hist.append(float(C))
        grad = np.asarray((sens @ B_ij[design_nodes])[design_nodes], dtype=float)
        return float(C), grad

    opt = optimization.Optimization(len(design_nodes), method="GD")
    opt.set_objective(objective)
    opt.set_maxiter(n_iter)
    opt.set_step_size(STEP_SIZE)
    opt.set_GD_normalized(True)
    x_opt = opt.optimize(sso.nodeparameters_values, log=False)

    zs[design_nodes] = x_opt
    zs_opt = zs.copy()
    zs_opt[design_nodes] = (B_ij @ zs)[design_nodes]

    sso.update_nodeparameter(jnp.array(zs_opt[design_nodes]))
    u = np.asarray(sso.params_u(sso.parameter_values, "sparse", True))
    u_nodal = u.reshape(n_node, 6)

    print(f"  [{label:30s}]  SE = {C_hist[-1]:.6e}  "
          f"({n_node} nodes, {len(cnct)} elems)")
    return zs_opt, u_nodal, C_hist


# =========================================================================
# 6. Run convergence study
# =========================================================================
results = {}
se_final = {}

def detect_boundary_nodes(xs, ys, cnct, cell_type="tri"):
    """Detect boundary nodes using pyvista's feature edge extraction."""
    from scipy.spatial import cKDTree
    pts3d = np.column_stack([xs, ys, np.zeros(len(xs))])
    n_cells = len(cnct)
    if cell_type == "quad":
        cells = np.hstack([np.full((n_cells, 1), 4, dtype=int), cnct])
        celltypes = np.full(n_cells, pv.CellType.QUAD)
    else:
        cells = np.hstack([np.full((n_cells, 1), 3, dtype=int), cnct])
        celltypes = np.full(n_cells, pv.CellType.TRIANGLE)
    grid = pv.UnstructuredGrid(cells, celltypes, pts3d)
    boundary = grid.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False
    )
    tree = cKDTree(pts3d)
    dists, idxs = tree.query(np.array(boundary.points))
    bc_flag = np.zeros(len(xs), dtype=int)
    bc_flag[idxs[dists < 1e-10]] = 1
    return bc_flag


for level in [0, 1]:
    print(f"\n--- Refinement level {level} ---")
    xs_l, ys_l, bc_l, cnct_tri_l, _ = build_level(level)
    B_ij = make_filter(xs_l, ys_l)
    n_iter = N_ITER.get(level, 400)

    # Tri shell at all levels
    key_tri = f"tri_L{level}"
    zs_opt, u_nodal, hist = run_opt(key_tri, xs_l, ys_l, bc_l, cnct_tri_l, "tri", B_ij, n_iter)
    results[key_tri] = (xs_l, ys_l, bc_l, cnct_tri_l, zs_opt, u_nodal, hist)
    se_final[key_tri] = hist[-1]

    # Beam on tri edges (homogenized) at all levels
    edges_l = extract_edges(cnct_tri_l)
    key_beam = f"beam_L{level}"
    zs_opt, u_nodal, hist = run_opt(key_beam, xs_l, ys_l, bc_l, edges_l, "beam", B_ij, n_iter)
    results[key_beam] = (xs_l, ys_l, bc_l, edges_l, zs_opt, u_nodal, hist)
    se_final[key_beam] = hist[-1]

    # Quad shell at all levels
    if level == 0:
        key_quad = "quad_L0"
        zs_opt, u_nodal, hist = run_opt(key_quad, xs_l, ys_l, bc_l, cnct_quad_orig, "quad", B_ij, n_iter)
        results[key_quad] = (xs_l, ys_l, bc_l, cnct_quad_orig, zs_opt, u_nodal, hist)
        se_final[key_quad] = hist[-1]
    elif level == 1:
        # Build refined quad mesh (different point set: adds face centers)
        xs_q, ys_q, cnct_quad_L1, _ = subdivide_quads(cnct_quad_orig, xs_orig, ys_orig)
        bc_q = detect_boundary_nodes(xs_q, ys_q, cnct_quad_L1, "quad")
        B_ij_q = make_filter(xs_q, ys_q)
        key_quad = "quad_L1"
        zs_opt, u_nodal, hist = run_opt(key_quad, xs_q, ys_q, bc_q, cnct_quad_L1, "quad", B_ij_q, n_iter)
        results[key_quad] = (xs_q, ys_q, bc_q, cnct_quad_L1, zs_opt, u_nodal, hist)
        se_final[key_quad] = hist[-1]


# =========================================================================
# 7. Export VTU files
# =========================================================================
print("\nExporting VTU files...")
for key, (xs_l, ys_l, bc_l, cnct_l, zs_opt, u_nodal, _) in results.items():
    pts = np.column_stack([xs_l, ys_l, zs_opt])
    n_cells = len(cnct_l)
    n_cols = cnct_l.shape[1] if cnct_l.ndim == 2 else 2

    if "quad" in key:
        cells = np.hstack([np.full((n_cells, 1), 4, dtype=int), cnct_l])
        celltypes = np.full(n_cells, pv.CellType.QUAD)
    elif "beam" in key:
        cells = np.hstack([np.full((n_cells, 1), 2, dtype=int), cnct_l])
        celltypes = np.full(n_cells, pv.CellType.LINE)
    else:
        cells = np.hstack([np.full((n_cells, 1), 3, dtype=int), cnct_l])
        celltypes = np.full(n_cells, pv.CellType.TRIANGLE)

    grid = pv.UnstructuredGrid(cells, celltypes, pts)
    grid.point_data["z_shape"] = zs_opt
    grid.point_data["displacement"] = u_nodal[:, :3]
    grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
    grid.point_data["bc_node"] = bc_l
    grid.save(str(OUT / f"{key}_opt.vtu"))

print("  Done.")

# Save results data for later plotting
data_to_save = {}
for key, (xs_l, ys_l, bc_l, cnct_l, zs_opt, u_nodal, hist) in results.items():
    data_to_save[key] = {
        "xs": xs_l, "ys": ys_l, "bc": bc_l, "cnct": cnct_l,
        "zs_opt": zs_opt, "u_nodal": u_nodal, "hist": np.array(hist),
    }
np.savez(str(OUT / "convergence_data.npz"), **{
    f"{key}__{field}": val
    for key, fields in data_to_save.items()
    for field, val in fields.items()
})
print(f"  Saved {OUT / 'convergence_data.npz'}")


# =========================================================================
# 8. Convergence plot: strain energy vs refinement
# =========================================================================
levels_all = sorted(set(int(k.split("_L")[1]) for k in results))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -- Panel 1: final strain energy vs level --
ax = axes[0]

for prefix, color, marker, label_fmt in [
    ("tri_", "C2", "o", "Tri shell (MITC3)"),
    ("quad_", "C0", "D", "Quad shell (MITC4)"),
    ("beam_", "C1", "s", "Beam grid (homog.)"),
]:
    keys = sorted([k for k in se_final if k.startswith(prefix)])
    if keys:
        lvls = [int(k.split("_L")[1]) for k in keys]
        ses = [se_final[k] for k in keys]
        ax.plot(lvls, ses, f"{marker}-", color=color, linewidth=2, markersize=8, label=label_fmt)

ax.set_xlabel("Refinement level", fontsize=13)
ax.set_ylabel("Final strain energy", fontsize=13)
ax.set_title("Strain energy convergence", fontsize=14)
ax.set_xticks(levels_all)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# -- Panel 2: optimization history --
ax = axes[1]
for key in sorted(results.keys()):
    hist = results[key][6]
    n_el = len(results[key][3])
    style = "--" if "quad" in key else "-"
    ax.plot(hist, style, linewidth=1.5, label=f"{key} ({n_el})")

ax.set_xlabel("Iteration", fontsize=13)
ax.set_ylabel("Strain energy", fontsize=13)
ax.set_yscale("log")
ax.set_title("Optimization history", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(OUT / "convergence_plot.png"), dpi=150, bbox_inches="tight")
print(f"  Saved {OUT / 'convergence_plot.png'}")


# =========================================================================
# 9. Shape comparison plots — top: shapes, bottom: errors vs reference
# =========================================================================

def get_tri_cnct_for_plot(key):
    """Get triangle connectivity suitable for matplotlib, regardless of element type."""
    xs_l, ys_l, bc_l, cnct_l, zs_opt, _, hist = results[key]
    if "quad" in key:
        return np.vstack([cnct_l[:, [0, 1, 2]], cnct_l[:, [0, 2, 3]]])
    elif "beam" in key:
        # Beam models share nodes with tri models at the same level;
        # use the tri connectivity for visualization
        level = key.split("_L")[1]
        tri_key = f"tri_L{level}"
        if tri_key in results:
            tri_cnct = results[tri_key][3]
        else:
            tri_cnct = cnct_tri_orig
        return tri_cnct
    else:
        return cnct_l

# Organize by level: columns = models at each level
level_models = {}
for key in sorted(results.keys()):
    level = int(key.split("_L")[1])
    level_models.setdefault(level, []).append(key)

# Sort within each level: quad, tri, beam
type_order = {"quad": 0, "tri": 1, "beam": 2}
for lvl in level_models:
    level_models[lvl].sort(key=lambda k: type_order.get(k.split("_L")[0], 9))

all_keys_flat = []
for lvl in sorted(level_models.keys()):
    all_keys_flat.extend(level_models[lvl])

n_cols = len(all_keys_flat)
fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))
if n_cols == 1:
    axes = axes.reshape(2, 1)

# Determine reference for errors at each level
# Use quad if available, else tri. Store (xs, ys, zs, ref_key) for interpolation.
ref_data = {}
for lvl in sorted(level_models.keys()):
    if f"quad_L{lvl}" in results:
        rk = f"quad_L{lvl}"
    elif f"tri_L{lvl}" in results:
        rk = f"tri_L{lvl}"
    else:
        continue
    ref_data[lvl] = (results[rk][0], results[rk][1], results[rk][4], rk)

# Global z range for consistent colorbar
all_z = np.concatenate([results[k][4] for k in all_keys_flat])
z_min, z_max = all_z.min(), all_z.max()

for col_idx, key in enumerate(all_keys_flat):
    xs_l, ys_l, bc_l, cnct_l, zs_opt, _, hist = results[key]
    tri_cnct = get_tri_cnct_for_plot(key)
    n_el = len(cnct_l)
    level = int(key.split("_L")[1])

    # Top row: optimized shape
    ax = axes[0, col_idx]
    sc = ax.tripcolor(xs_l, ys_l, tri_cnct, zs_opt, cmap="viridis",
                      shading="gouraud", vmin=z_min, vmax=z_max)
    ax.set_aspect("equal")
    ax.set_title(f"{key}\n{len(xs_l)} nodes, {n_el} elems\nSE = {hist[-1]:.1f}", fontsize=10)
    ax.axis("off")
    plt.colorbar(sc, ax=ax, label="z", shrink=0.6)

    # Bottom row: error vs reference at this level
    ax = axes[1, col_idx]
    rd = ref_data.get(level)
    ref_key = rd[3] if rd else None
    if rd is not None and key != ref_key:
        xs_ref, ys_ref, zs_ref = rd[0], rd[1], rd[2]
        if len(zs_ref) == len(zs_opt):
            err = zs_opt - zs_ref
        else:
            # Interpolate reference onto this model's nodes
            from scipy.interpolate import LinearNDInterpolator
            interp = LinearNDInterpolator(np.column_stack([xs_ref, ys_ref]), zs_ref)
            zs_ref_interp = interp(xs_l, ys_l)
            zs_ref_interp = np.nan_to_num(zs_ref_interp, nan=0.0)
            err = zs_opt - zs_ref_interp
        elim = max(abs(err.min()), abs(err.max()), 0.1)
        sc = ax.tripcolor(xs_l, ys_l, tri_cnct, err, cmap="RdBu_r",
                          shading="gouraud", vmin=-elim, vmax=elim)
        ax.set_title(f"z error vs {ref_key}\nRMS={np.sqrt(np.mean(err**2)):.3f}", fontsize=10)
        plt.colorbar(sc, ax=ax, label="dz", shrink=0.6)
    else:
        ax.text(0.5, 0.5, "reference", ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
        ax.set_title("(reference)", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

fig.suptitle("Optimized shapes and errors", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(str(OUT / "shapes_comparison.png"), dpi=150, bbox_inches="tight")
print(f"  Saved {OUT / 'shapes_comparison.png'}")


# =========================================================================
# 10. Summary table
# =========================================================================
print("\n" + "=" * 70)
print("CONVERGENCE SUMMARY")
print("=" * 70)
print(f"{'Model':<20s} {'Nodes':>8s} {'Elements':>10s} {'Final SE':>14s} {'Ratio vs L0 tri':>16s}")
print("-" * 70)
se_ref = se_final["tri_L0"]
for key in sorted(results.keys()):
    xs_l, ys_l, bc_l, cnct_l, _, _, hist = results[key]
    se = hist[-1]
    print(f"  {key:<18s} {len(xs_l):8d} {len(cnct_l):10d} {se:14.4e} {se/se_ref:16.4f}")
