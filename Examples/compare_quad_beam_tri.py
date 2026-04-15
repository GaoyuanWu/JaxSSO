"""
Shape optimization comparison on the Mannheim Multihalle mesh.

Models
------
1. Quad shell  (MITC4)               — true material properties
2. Tri shell   (MITC3)               — true material properties (triangulated quad mesh)
3. Beam grid on quad edges            — inverse-homogenized from shell
4. Beam grid on triangle edges        — inverse-homogenized from shell

For each model we export:
  - base (undeformed) mesh as VTU
  - optimized mesh as VTU with displacement and shape fields

A difference VTU compares quad-shell vs each beam model point-wise.

Inverse homogenization formulae
-------------------------------
Quad grid (2 bar families at 0/90 deg, spacing d):
    A  = t * d           (match membrane Et)
    I  = t^3 * d / 12    (match bending D = Et^3/12)
    nu_eff = 0, no shear coupling

Tri grid (3 bar families at 0/60/120 deg, effective spacing d):
    A  = sqrt(3)/2 * t * d    (match isotropic membrane, nu_eff = 1/3)
    I  = sqrt(3)/2 * t^3 * d / 12   (match bending)
"""

import numpy as np
import jax
import jax.numpy as jnp
import pyvista as pv
from pathlib import Path

jax.config.update("jax_enable_x64", True)

import JaxSSO.model as Model
import JaxSSO.optimization as optimization
from JaxSSO.SSO_model import NodeParameter, SSO_model

OUT = Path("Examples/output")
OUT.mkdir(exist_ok=True)

# =========================================================================
# 1. Load mesh from VTU
# =========================================================================
mesh_path = Path("Examples/Data/mannheim_quad.vtu")
if not mesh_path.exists():
    raise FileNotFoundError(
        f"{mesh_path} not found. Run convert_mesh_to_vtu.py first."
    )

base_mesh = pv.read(str(mesh_path))
raw_points = np.array(base_mesh.points)
bc_flag = np.array(base_mesh.point_data["bc_node"])

n_node = base_mesh.n_points
n_ele_quad = base_mesh.n_cells
cnct_quad = base_mesh.cells.reshape(-1, 5)[:, 1:]  # drop the leading '4'

# Normalize xy (keep z as-is for initial shape later)
xs = raw_points[:, 0] - raw_points[:, 0].min()
ys = raw_points[:, 1] - raw_points[:, 1].min()

bc_nodes = np.where(bc_flag == 1)[0]
design_nodes = np.where(bc_flag == 0)[0]

print(f"Mesh loaded: {n_node} nodes, {n_ele_quad} quads")
print(f"  design nodes: {len(design_nodes)},  BC nodes: {len(bc_nodes)}")

# =========================================================================
# 2. Derive topologies
# =========================================================================
# Quad edges
quad_edge_set = set()
for q in cnct_quad:
    for a, b in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]:
        quad_edge_set.add((min(a, b), max(a, b)))
cnct_quad_bar = np.array(sorted(quad_edge_set), dtype=int)

# Triangulate
pts_for_tri = np.column_stack([xs, ys, np.zeros(n_node)])  # flat for topology only
quad_pv = pv.UnstructuredGrid(
    np.hstack([np.full((n_ele_quad, 1), 4, dtype=int), cnct_quad]),
    np.full(n_ele_quad, pv.CellType.QUAD),
    pts_for_tri,
)
tri_pv = quad_pv.triangulate()
cnct_tri = tri_pv.cells.reshape(-1, 4)[:, 1:]
n_ele_tri = cnct_tri.shape[0]

# Triangle edges
tri_edge_set = set()
for tri in cnct_tri:
    for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
        tri_edge_set.add((min(a, b), max(a, b)))
cnct_tri_bar = np.array(sorted(tri_edge_set), dtype=int)

print(f"  quad edges: {len(cnct_quad_bar)},  triangles: {n_ele_tri},  tri edges: {len(cnct_tri_bar)}")

# =========================================================================
# 3. Material properties and filter
# =========================================================================
t = 0.1       # shell thickness [m]
E = 1e10      # Young's modulus [Pa]
nu = 0.3      # Poisson's ratio
load = 5000.0  # vertical load per free node [N]
G = E / (2 * (1 + nu))

N_ITER = 200
STEP_SIZE = 1e-1

# Linear hat filter (smooths design variables)
def linear_filter(dist, R):
    return np.where(dist > R, 0.0, (1.0 / R) * (R - dist))

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
D = np.sqrt(dx**2 + dy**2)
B_ij_raw = linear_filter(D, 10.0)
B_ij = B_ij_raw / B_ij_raw.sum(axis=1, keepdims=True)

# =========================================================================
# 4. Inverse homogenization: shell -> beam properties
# =========================================================================
def avg_edge_length(pts, edges):
    return np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1).mean()


def beam_props_quad_homog(E, t, d):
    """Inverse homogenization for a quad grid (2 bar families, 0/90 deg)."""
    A = t * d
    Iy = t**3 * d / 12
    Iz = Iy
    J = Iy + Iz
    G_b = E / (2 * (1 + 0.0))  # nu_eff = 0 for quad grid
    return A, Iy, Iz, J, G_b


def beam_props_tri_homog(E, t, d):
    """Inverse homogenization for a tri grid (3 bar families, 0/60/120 deg)."""
    s3h = np.sqrt(3) / 2
    A = s3h * t * d
    Iy = s3h * t**3 * d / 12
    Iz = Iy
    J = Iy + Iz
    G_b = E / (2 * (1 + 1 / 3))  # nu_eff = 1/3 for tri grid
    return A, Iy, Iz, J, G_b


# =========================================================================
# 5. Shape optimization driver
# =========================================================================
def run_shape_opt(label, build_model_fn):
    """
    Run shape optimization.

    Parameters
    ----------
    label : str
        Name for this run (used in output filenames).
    build_model_fn : callable(zs_filtered) -> Model
        Function that builds a JaxSSO Model given filtered z-coordinates.

    Returns
    -------
    zs_init, zs_opt : (n_node,) arrays
        Initial and optimized z-coordinates (filtered).
    u_opt : (n_node, 6) array
        Nodal displacements at the optimized configuration.
    hist : list of float
        Strain energy history.
    """
    # Reproducible random initial z for design nodes
    rng = np.random.RandomState(42)
    zs = np.zeros(n_node)
    zs[design_nodes] = rng.uniform(0.5, 0.51, size=len(design_nodes))

    zs_filt = zs.copy()
    zs_filt[design_nodes] = (B_ij @ zs_filt)[design_nodes]
    zs_init = zs_filt.copy()

    # Build FE model
    model = build_model_fn(zs_filt)
    model.model_ready()

    # SSO model
    sso = SSO_model(model)
    for nd in design_nodes:
        sso.add_nodeparameter(NodeParameter(nd, 2))
    sso.initialize_parameters_values()
    sso.set_objective(objective="strain energy")

    # Filtered objective
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

    # Optimize
    opt = optimization.Optimization(len(design_nodes), method="GD")
    opt.set_objective(objective)
    opt.set_maxiter(N_ITER)
    opt.set_step_size(STEP_SIZE)
    opt.set_GD_normalized(True)
    x_opt = opt.optimize(sso.nodeparameters_values, log=False)

    # Recover optimized z
    zs[design_nodes] = x_opt
    zs_opt = zs.copy()
    zs_opt[design_nodes] = (B_ij @ zs)[design_nodes]

    # Get displacement at optimized config
    sso.update_nodeparameter(jnp.array(zs_opt[design_nodes]))
    u = np.asarray(sso.params_u(sso.parameter_values, "sparse", True))
    u_nodal = u.reshape(n_node, 6)

    print(f"  [{label:20s}]  SE = {C_hist[-1]:.6e}  ({len(C_hist)} iters)")
    return zs_init, zs_opt, u_nodal, C_hist


# =========================================================================
# 6. Model builders
# =========================================================================
def make_quad_shell(zs_filt):
    model = Model.Model()
    for i in range(n_node):
        model.add_node(i, xs[i], ys[i], zs_filt[i])
        if bc_flag[i]:
            model.add_support(i, [1, 1, 1, 1, 1, 1])
        else:
            model.add_nodal_load(i, nodal_load=[0, 0, -load, 0, 0, 0])
    for i in range(n_ele_quad):
        model.add_quad(i, *cnct_quad[i], t, E, nu)
    return model


def _make_beam_model(zs_filt, edges, A, Iy, Iz, J, G_b):
    model = Model.Model()
    for i in range(n_node):
        model.add_node(i, xs[i], ys[i], zs_filt[i])
        if bc_flag[i]:
            model.add_support(i, [1, 1, 1, 1, 1, 1])
        else:
            model.add_nodal_load(i, nodal_load=[0, 0, -load, 0, 0, 0])
    for i in range(len(edges)):
        model.add_beamcol(i, edges[i, 0], edges[i, 1], E, G_b, Iy, Iz, J, A)
    return model


# Compute average edge lengths for homogenization
pts_flat = np.column_stack([xs, ys, np.zeros(n_node)])
d_quad = avg_edge_length(pts_flat, cnct_quad_bar)
d_tri = avg_edge_length(pts_flat, cnct_tri_bar)

A_q, Iy_q, Iz_q, J_q, G_q = beam_props_quad_homog(E, t, d_quad)
A_t, Iy_t, Iz_t, J_t, G_t = beam_props_tri_homog(E, t, d_tri)

print(f"\nHomogenized beam properties:")
print(f"  Quad grid (d={d_quad:.3f}):  A={A_q:.4e}  I={Iy_q:.4e}")
print(f"  Tri  grid (d={d_tri:.3f}):   A={A_t:.4e}  I={Iy_t:.4e}")


def make_tri_shell(zs_filt):
    model = Model.Model()
    for i in range(n_node):
        model.add_node(i, xs[i], ys[i], zs_filt[i])
        if bc_flag[i]:
            model.add_support(i, [1, 1, 1, 1, 1, 1])
        else:
            model.add_nodal_load(i, nodal_load=[0, 0, -load, 0, 0, 0])
    for i in range(n_ele_tri):
        model.add_tri(i, cnct_tri[i, 0], cnct_tri[i, 1], cnct_tri[i, 2], t, E, nu)
    return model


def make_beam_quad(zs_filt):
    return _make_beam_model(zs_filt, cnct_quad_bar, A_q, Iy_q, Iz_q, J_q, G_q)


def make_beam_tri(zs_filt):
    return _make_beam_model(zs_filt, cnct_tri_bar, A_t, Iy_t, Iz_t, J_t, G_t)


# =========================================================================
# 7. Run all optimizations
# =========================================================================
runs = {
    "quad_shell": make_quad_shell,
    "tri_shell": make_tri_shell,
    "beam_quad_homog": make_beam_quad,
    "beam_tri_homog": make_beam_tri,
}

results = {}
for label, builder in runs.items():
    print(f"\nRunning: {label}")
    results[label] = run_shape_opt(label, builder)


# =========================================================================
# 8. Export VTU files
# =========================================================================
def save_quad_vtu(name, zs_vals, u_nodal=None):
    pts = np.column_stack([xs, ys, zs_vals])
    cells = np.hstack([np.full((n_ele_quad, 1), 4, dtype=int), cnct_quad])
    grid = pv.UnstructuredGrid(cells, np.full(n_ele_quad, pv.CellType.QUAD), pts)
    grid.point_data["z_shape"] = zs_vals
    grid.point_data["bc_node"] = bc_flag
    if u_nodal is not None:
        grid.point_data["displacement"] = u_nodal[:, :3]
        grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
        grid.point_data["disp_z"] = u_nodal[:, 2]
    path = OUT / f"{name}.vtu"
    grid.save(str(path))
    return grid


def save_tri_vtu(name, zs_vals, u_nodal=None):
    pts = np.column_stack([xs, ys, zs_vals])
    cells = np.hstack([np.full((n_ele_tri, 1), 3, dtype=int), cnct_tri])
    grid = pv.UnstructuredGrid(cells, np.full(n_ele_tri, pv.CellType.TRIANGLE), pts)
    grid.point_data["z_shape"] = zs_vals
    grid.point_data["bc_node"] = bc_flag
    if u_nodal is not None:
        grid.point_data["displacement"] = u_nodal[:, :3]
        grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
        grid.point_data["disp_z"] = u_nodal[:, 2]
    path = OUT / f"{name}.vtu"
    grid.save(str(path))
    return grid


def save_line_vtu(name, edges, zs_vals, u_nodal=None):
    pts = np.column_stack([xs, ys, zs_vals])
    cells = np.hstack([np.full((len(edges), 1), 2, dtype=int), edges])
    grid = pv.UnstructuredGrid(cells, np.full(len(edges), pv.CellType.LINE), pts)
    grid.point_data["z_shape"] = zs_vals
    grid.point_data["bc_node"] = bc_flag
    if u_nodal is not None:
        grid.point_data["displacement"] = u_nodal[:, :3]
        grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
        grid.point_data["disp_z"] = u_nodal[:, 2]
    path = OUT / f"{name}.vtu"
    grid.save(str(path))
    return grid


print("\nExporting VTU files to Examples/output/...")

# ---- Helpers for deformed meshes ----
def save_quad_deformed_vtu(name, zs_vals, u_nodal):
    """Save the deformed configuration: geometry = optimized shape + displacement."""
    pts_opt = np.column_stack([xs, ys, zs_vals])
    pts_def = pts_opt + u_nodal[:, :3]
    cells = np.hstack([np.full((n_ele_quad, 1), 4, dtype=int), cnct_quad])
    grid = pv.UnstructuredGrid(cells, np.full(n_ele_quad, pv.CellType.QUAD), pts_def)
    grid.point_data["displacement"] = u_nodal[:, :3]
    grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
    grid.point_data["z_undeformed"] = zs_vals
    grid.point_data["z_deformed"] = pts_def[:, 2]
    grid.point_data["bc_node"] = bc_flag
    path = OUT / f"{name}.vtu"
    grid.save(str(path))
    return grid


def save_tri_deformed_vtu(name, zs_vals, u_nodal):
    """Save the deformed configuration for a tri shell."""
    pts_opt = np.column_stack([xs, ys, zs_vals])
    pts_def = pts_opt + u_nodal[:, :3]
    cells = np.hstack([np.full((n_ele_tri, 1), 3, dtype=int), cnct_tri])
    grid = pv.UnstructuredGrid(cells, np.full(n_ele_tri, pv.CellType.TRIANGLE), pts_def)
    grid.point_data["displacement"] = u_nodal[:, :3]
    grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
    grid.point_data["z_undeformed"] = zs_vals
    grid.point_data["z_deformed"] = pts_def[:, 2]
    grid.point_data["bc_node"] = bc_flag
    path = OUT / f"{name}.vtu"
    grid.save(str(path))
    return grid


def save_line_deformed_vtu(name, edges, zs_vals, u_nodal):
    """Save the deformed configuration for a beam grid."""
    pts_opt = np.column_stack([xs, ys, zs_vals])
    pts_def = pts_opt + u_nodal[:, :3]
    cells = np.hstack([np.full((len(edges), 1), 2, dtype=int), edges])
    grid = pv.UnstructuredGrid(cells, np.full(len(edges), pv.CellType.LINE), pts_def)
    grid.point_data["displacement"] = u_nodal[:, :3]
    grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
    grid.point_data["z_undeformed"] = zs_vals
    grid.point_data["z_deformed"] = pts_def[:, 2]
    grid.point_data["bc_node"] = bc_flag
    path = OUT / f"{name}.vtu"
    grid.save(str(path))
    return grid


# ---- Export each model: base, optimized (unloaded), deformed (loaded) ----

# -- Quad shell --
zs_init_qs, zs_opt_qs, u_qs, _ = results["quad_shell"]
save_quad_vtu("quad_shell_base", zs_init_qs)
save_quad_vtu("quad_shell_opt", zs_opt_qs, u_qs)
save_quad_deformed_vtu("quad_shell_deformed", zs_opt_qs, u_qs)

# -- Tri shell --
zs_init_ts, zs_opt_ts, u_ts, _ = results["tri_shell"]
save_tri_vtu("tri_shell_base", zs_init_ts)
save_tri_vtu("tri_shell_opt", zs_opt_ts, u_ts)
save_tri_deformed_vtu("tri_shell_deformed", zs_opt_ts, u_ts)

# -- Beam quad grid (homogenized) --
zs_init_bq, zs_opt_bq, u_bq, _ = results["beam_quad_homog"]
save_line_vtu("beam_quad_homog_base", cnct_quad_bar, zs_init_bq)
save_line_vtu("beam_quad_homog_opt", cnct_quad_bar, zs_opt_bq, u_bq)
save_line_deformed_vtu("beam_quad_homog_deformed", cnct_quad_bar, zs_opt_bq, u_bq)

# -- Beam tri grid (homogenized) --
zs_init_bt, zs_opt_bt, u_bt, _ = results["beam_tri_homog"]
save_line_vtu("beam_tri_homog_base", cnct_tri_bar, zs_init_bt)
save_line_vtu("beam_tri_homog_opt", cnct_tri_bar, zs_opt_bt, u_bt)
save_line_deformed_vtu("beam_tri_homog_deformed", cnct_tri_bar, zs_opt_bt, u_bt)

# ---- Difference meshes (quad shell as reference) ----
# Compare both the optimized shape (SO result) and the deformed config (FEA result)
for other_label in ["tri_shell", "beam_quad_homog", "beam_tri_homog"]:
    _, zs_opt_o, u_o, _ = results[other_label]

    # Difference in optimized shape (unloaded)
    diff_z = zs_opt_qs - zs_opt_o

    # Difference in displacement field
    diff_u = u_qs[:, :3] - u_o[:, :3]

    # Difference in deformed position (shape + displacement)
    pos_qs = np.column_stack([xs, ys, zs_opt_qs]) + u_qs[:, :3]
    pos_o = np.column_stack([xs, ys, zs_opt_o]) + u_o[:, :3]
    diff_deformed = pos_qs - pos_o

    # Write on the quad-shell optimized geometry for visualization
    pts = np.column_stack([xs, ys, zs_opt_qs])
    cells = np.hstack([np.full((n_ele_quad, 1), 4, dtype=int), cnct_quad])
    grid = pv.UnstructuredGrid(cells, np.full(n_ele_quad, pv.CellType.QUAD), pts)

    grid.point_data["diff_z_shape"] = diff_z
    grid.point_data["diff_z_shape_abs"] = np.abs(diff_z)
    grid.point_data["diff_displacement"] = diff_u
    grid.point_data["diff_disp_magnitude"] = np.linalg.norm(diff_u, axis=1)
    grid.point_data["diff_deformed_pos"] = diff_deformed
    grid.point_data["diff_deformed_magnitude"] = np.linalg.norm(diff_deformed, axis=1)
    grid.save(str(OUT / f"diff_quad_shell_vs_{other_label}.vtu"))

print("  Done.")

# =========================================================================
# 9. Summary
# =========================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Model':<25s} {'Final SE':>14s}  {'Ratio vs quad':>14s}")
print("-" * 55)
se_ref = results["quad_shell"][3][-1]
for label in runs:
    se = results[label][3][-1]
    ratio = se / se_ref
    print(f"  {label:<23s} {se:14.6e}  {ratio:14.4f}")

others = ["tri_shell", "beam_quad_homog", "beam_tri_homog"]

print(f"\n--- Shape optimization result (unloaded geometry) ---")
print(f"{'Comparison':<45s} {'RMS dz':>10s} {'max |dz|':>10s}")
print("-" * 67)
for label in others:
    dz = zs_opt_qs - results[label][1]
    print(f"  quad_shell vs {label:<27s} {np.sqrt(np.mean(dz**2)):10.4f} {np.max(np.abs(dz)):10.4f}")

print(f"\n--- Displacement field (FEA at optimized shape) ---")
print(f"{'Comparison':<45s} {'RMS |du|':>10s} {'max |du|':>10s}")
print("-" * 67)
for label in others:
    du = u_qs[:, :3] - results[label][2][:, :3]
    rms = np.sqrt(np.mean(np.sum(du**2, axis=1)))
    mx = np.max(np.linalg.norm(du, axis=1))
    print(f"  quad_shell vs {label:<27s} {rms:10.4e} {mx:10.4e}")

print(f"\n--- Deformed configuration (shape + displacement) ---")
print(f"{'Comparison':<45s} {'RMS |dp|':>10s} {'max |dp|':>10s}")
print("-" * 67)
for label in others:
    _, zs_o, u_o, _ = results[label]
    pos_qs = np.column_stack([xs, ys, zs_opt_qs]) + u_qs[:, :3]
    pos_o = np.column_stack([xs, ys, zs_o]) + u_o[:, :3]
    dp = pos_qs - pos_o
    rms = np.sqrt(np.mean(np.sum(dp**2, axis=1)))
    mx = np.max(np.linalg.norm(dp, axis=1))
    print(f"  quad_shell vs {label:<27s} {rms:10.4f} {mx:10.4f}")

print(f"\nFiles written to {OUT}/:")
for f in sorted(OUT.glob("*.vtu")):
    print(f"  {f.name}")
