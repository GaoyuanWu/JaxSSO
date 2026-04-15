"""
Combined example runner for JaxSSO shape optimization on the Mannheim Multihalle mesh.

Three studies:
  1. run_comparison()  — Quad/Tri shell vs Beam grid comparison (MITC4, MITC3, homogenized beams)
  2. run_convergence() — Mesh refinement convergence study (levels 0 and 1)
  3. run_benchmark()   — Performance benchmark (JIT time, per-iteration time, memory)

All outputs go to Examples/output/<study>/.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pyvista as pv
import matplotlib.pyplot as plt
import time
import tracemalloc
from pathlib import Path

jax.config.update("jax_enable_x64", True)

import JaxSSO.model as Model
import JaxSSO.optimization as optimization
from JaxSSO.SSO_model import NodeParameter, SSO_model

# =========================================================================
# Material constants
# =========================================================================
t = 0.1        # shell thickness [m]
E = 1e10       # Young's modulus [Pa]
nu = 0.3       # Poisson's ratio
load = 5000.0  # vertical load per free node [N]

# =========================================================================
# Shared utility functions
# =========================================================================

def load_mesh():
    """Load the Mannheim quad mesh from VTU and return basic arrays."""
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
    cnct_quad = base_mesh.cells.reshape(-1, 5)[:, 1:]
    xs = raw_points[:, 0] - raw_points[:, 0].min()
    ys = raw_points[:, 1] - raw_points[:, 1].min()
    return xs, ys, bc_flag, cnct_quad, n_node, n_ele_quad


def triangulate_short_diagonal(cnct_quad, xs, ys):
    """Split each quad into 2 triangles along the shorter diagonal."""
    tris = []
    for q in cnct_quad:
        n0, n1, n2, n3 = q
        d02 = np.sqrt((xs[n2] - xs[n0])**2 + (ys[n2] - ys[n0])**2)
        d13 = np.sqrt((xs[n3] - xs[n1])**2 + (ys[n3] - ys[n1])**2)
        if d02 <= d13:
            tris.append([n0, n1, n2])
            tris.append([n0, n2, n3])
        else:
            tris.append([n0, n1, n3])
            tris.append([n1, n2, n3])
    return np.array(tris, dtype=int)


def subdivide_quads(cnct_quad, xs, ys):
    """
    Subdivide each quad into 4 quads by adding edge midpoints and face centers.
    Returns (xs_new, ys_new, cnct_new, edge_to_mid) with new nodes appended.
    """
    n_node = len(xs)
    xs_new = list(xs)
    ys_new = list(ys)
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
    n_after_edges = len(xs_new)
    cnct_new = []
    for i, q in enumerate(cnct_quad):
        n0, n1, n2, n3 = q
        fc_id = n_after_edges + i
        xs_new.append(0.25 * (xs[n0] + xs[n1] + xs[n2] + xs[n3]))
        ys_new.append(0.25 * (ys[n0] + ys[n1] + ys[n2] + ys[n3]))
        m01 = edge_to_mid[(min(n0, n1), max(n0, n1))]
        m12 = edge_to_mid[(min(n1, n2), max(n1, n2))]
        m23 = edge_to_mid[(min(n2, n3), max(n2, n3))]
        m30 = edge_to_mid[(min(n3, n0), max(n3, n0))]
        cnct_new.append([n0, m01, fc_id, m30])
        cnct_new.append([m01, n1, m12, fc_id])
        cnct_new.append([fc_id, m12, n2, m23])
        cnct_new.append([m30, fc_id, m23, n3])
    return np.array(xs_new), np.array(ys_new), np.array(cnct_new, dtype=int), edge_to_mid


def extract_edges(cnct_tri):
    """Extract unique edges from triangle connectivity."""
    edge_set = set()
    for f in cnct_tri:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            edge_set.add((min(a, b), max(a, b)))
    return np.array(sorted(edge_set), dtype=int)


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


def make_filter(xs, ys, R=10.0):
    """Linear hat filter matrix (smooths design variables)."""
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D = np.sqrt(dx**2 + dy**2)
    B_raw = np.where(D > R, 0.0, (1.0 / R) * (R - D))
    return B_raw / B_raw.sum(axis=1, keepdims=True)


def avg_edge_length(pts, edges):
    """Average edge length in a mesh."""
    return np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1).mean()


###########################################################################
#
#  1. COMPARISON STUDY
#
###########################################################################

def run_comparison(n_iter=200, step_size=1e-1, out=Path("Examples/output/comparison")):
    """
    Compare quad shell (MITC4), tri shell (MITC3), beam-on-quad-edges,
    and beam-on-tri-edges for shape optimization on the Mannheim mesh.

    Runs 4 shape optimizations on the same L0 mesh with different element
    types, exports VTU meshes (base, optimized, deformed, difference),
    and prints a comparison summary.

    Parameters
    ----------
    n_iter : int
        Number of gradient descent iterations per model.
    step_size : float
        Step size for normalized gradient descent.
    """
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load mesh
    # -----------------------------------------------------------------
    xs, ys, bc_flag, cnct_quad, n_node, n_ele_quad = load_mesh()
    bc_nodes = np.where(bc_flag == 1)[0]
    design_nodes = np.where(bc_flag == 0)[0]

    print(f"Mesh loaded: {n_node} nodes, {n_ele_quad} quads")
    print(f"  design nodes: {len(design_nodes)},  BC nodes: {len(bc_nodes)}")

    # -----------------------------------------------------------------
    # Derive topologies
    # -----------------------------------------------------------------
    # Quad edges
    quad_edge_set = set()
    for q in cnct_quad:
        for a, b in [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]:
            quad_edge_set.add((min(a, b), max(a, b)))
    cnct_quad_bar = np.array(sorted(quad_edge_set), dtype=int)

    # Triangulate
    pts_for_tri = np.column_stack([xs, ys, np.zeros(n_node)])
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

    # -----------------------------------------------------------------
    # Filter
    # -----------------------------------------------------------------
    G = E / (2 * (1 + nu))
    N_ITER = n_iter
    STEP_SIZE = step_size
    B_ij = make_filter(xs, ys)

    # -----------------------------------------------------------------
    # Homogenized beam properties
    # -----------------------------------------------------------------
    pts_flat = np.column_stack([xs, ys, np.zeros(n_node)])
    d_quad = avg_edge_length(pts_flat, cnct_quad_bar)
    d_tri = avg_edge_length(pts_flat, cnct_tri_bar)

    A_q, Iy_q, Iz_q, J_q, G_q = beam_props_quad_homog(E, t, d_quad)
    A_t, Iy_t, Iz_t, J_t, G_t = beam_props_tri_homog(E, t, d_tri)

    print(f"\nHomogenized beam properties:")
    print(f"  Quad grid (d={d_quad:.3f}):  A={A_q:.4e}  I={Iy_q:.4e}")
    print(f"  Tri  grid (d={d_tri:.3f}):   A={A_t:.4e}  I={Iy_t:.4e}")

    # -----------------------------------------------------------------
    # Shape optimization driver (local to comparison)
    # -----------------------------------------------------------------
    def run_shape_opt(label, build_model_fn):
        rng = np.random.RandomState(42)
        zs = np.zeros(n_node)
        zs[design_nodes] = rng.uniform(0.5, 0.51, size=len(design_nodes))
        zs_filt = zs.copy()
        zs_filt[design_nodes] = (B_ij @ zs_filt)[design_nodes]
        zs_init = zs_filt.copy()

        model = build_model_fn(zs_filt)
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
        opt.set_maxiter(N_ITER)
        opt.set_step_size(STEP_SIZE)
        opt.set_GD_normalized(True)
        x_opt = opt.optimize(sso.nodeparameters_values, log=False)

        zs[design_nodes] = x_opt
        zs_opt = zs.copy()
        zs_opt[design_nodes] = (B_ij @ zs)[design_nodes]

        sso.update_nodeparameter(jnp.array(zs_opt[design_nodes]))
        u = np.asarray(sso.params_u(sso.parameter_values, "sparse", True))
        u_nodal = u.reshape(n_node, 6)

        print(f"  [{label:20s}]  SE = {C_hist[-1]:.6e}  ({len(C_hist)} iters)")
        return zs_init, zs_opt, u_nodal, C_hist

    # -----------------------------------------------------------------
    # Model builders
    # -----------------------------------------------------------------
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

    def make_beam_quad(zs_filt):
        return _make_beam_model(zs_filt, cnct_quad_bar, A_q, Iy_q, Iz_q, J_q, G_q)

    def make_beam_tri(zs_filt):
        return _make_beam_model(zs_filt, cnct_tri_bar, A_t, Iy_t, Iz_t, J_t, G_t)

    # -----------------------------------------------------------------
    # Run all optimizations
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Export VTU files
    # -----------------------------------------------------------------
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
        path = out / f"{name}.vtu"
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
        path = out / f"{name}.vtu"
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
        path = out / f"{name}.vtu"
        grid.save(str(path))
        return grid

    def save_quad_deformed_vtu(name, zs_vals, u_nodal):
        pts_opt = np.column_stack([xs, ys, zs_vals])
        pts_def = pts_opt + u_nodal[:, :3]
        cells = np.hstack([np.full((n_ele_quad, 1), 4, dtype=int), cnct_quad])
        grid = pv.UnstructuredGrid(cells, np.full(n_ele_quad, pv.CellType.QUAD), pts_def)
        grid.point_data["displacement"] = u_nodal[:, :3]
        grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
        grid.point_data["z_undeformed"] = zs_vals
        grid.point_data["z_deformed"] = pts_def[:, 2]
        grid.point_data["bc_node"] = bc_flag
        path = out / f"{name}.vtu"
        grid.save(str(path))
        return grid

    def save_tri_deformed_vtu(name, zs_vals, u_nodal):
        pts_opt = np.column_stack([xs, ys, zs_vals])
        pts_def = pts_opt + u_nodal[:, :3]
        cells = np.hstack([np.full((n_ele_tri, 1), 3, dtype=int), cnct_tri])
        grid = pv.UnstructuredGrid(cells, np.full(n_ele_tri, pv.CellType.TRIANGLE), pts_def)
        grid.point_data["displacement"] = u_nodal[:, :3]
        grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
        grid.point_data["z_undeformed"] = zs_vals
        grid.point_data["z_deformed"] = pts_def[:, 2]
        grid.point_data["bc_node"] = bc_flag
        path = out / f"{name}.vtu"
        grid.save(str(path))
        return grid

    def save_line_deformed_vtu(name, edges, zs_vals, u_nodal):
        pts_opt = np.column_stack([xs, ys, zs_vals])
        pts_def = pts_opt + u_nodal[:, :3]
        cells = np.hstack([np.full((len(edges), 1), 2, dtype=int), edges])
        grid = pv.UnstructuredGrid(cells, np.full(len(edges), pv.CellType.LINE), pts_def)
        grid.point_data["displacement"] = u_nodal[:, :3]
        grid.point_data["disp_magnitude"] = np.linalg.norm(u_nodal[:, :3], axis=1)
        grid.point_data["z_undeformed"] = zs_vals
        grid.point_data["z_deformed"] = pts_def[:, 2]
        grid.point_data["bc_node"] = bc_flag
        path = out / f"{name}.vtu"
        grid.save(str(path))
        return grid

    print("\nExporting VTU files to Examples/output/comparison/...")

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

    # -- Difference meshes (quad shell as reference) --
    for other_label in ["tri_shell", "beam_quad_homog", "beam_tri_homog"]:
        _, zs_opt_o, u_o, _ = results[other_label]
        diff_z = zs_opt_qs - zs_opt_o
        diff_u = u_qs[:, :3] - u_o[:, :3]
        pos_qs = np.column_stack([xs, ys, zs_opt_qs]) + u_qs[:, :3]
        pos_o = np.column_stack([xs, ys, zs_opt_o]) + u_o[:, :3]
        diff_deformed = pos_qs - pos_o

        pts = np.column_stack([xs, ys, zs_opt_qs])
        cells = np.hstack([np.full((n_ele_quad, 1), 4, dtype=int), cnct_quad])
        grid = pv.UnstructuredGrid(cells, np.full(n_ele_quad, pv.CellType.QUAD), pts)
        grid.point_data["diff_z_shape"] = diff_z
        grid.point_data["diff_z_shape_abs"] = np.abs(diff_z)
        grid.point_data["diff_displacement"] = diff_u
        grid.point_data["diff_disp_magnitude"] = np.linalg.norm(diff_u, axis=1)
        grid.point_data["diff_deformed_pos"] = diff_deformed
        grid.point_data["diff_deformed_magnitude"] = np.linalg.norm(diff_deformed, axis=1)
        grid.save(str(out / f"diff_quad_shell_vs_{other_label}.vtu"))

    print("  Done.")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
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

    print(f"\nFiles written to {out}/:")
    for f in sorted(out.glob("*.vtu")):
        print(f"  {f.name}")


###########################################################################
#
#  2. CONVERGENCE STUDY
#
###########################################################################

def run_convergence(levels=(0, 1), n_iter_per_level=None, step_size=5e-2, out=Path("Examples/output/convergence")):
    """
    Convergence study: quad shell (MITC4), tri shell (MITC3), and beam grid
    (homogenized) on the Mannheim mesh at multiple refinement levels.

    Subdivides the mesh using pyvista (tri) and midpoint subdivision (quad),
    runs shape optimization at each level, and produces convergence plots,
    shape comparison figures, and VTU exports.

    Parameters
    ----------
    levels : tuple of int
        Refinement levels to run. Level 0 = original mesh, level 1 = one
        subdivision (4x elements).
    n_iter_per_level : dict or None
        Map from level to number of GD iterations. Defaults to
        {0: 400, 1: 600} if None.
    step_size : float
        Step size for normalized gradient descent.
    """
    out.mkdir(parents=True, exist_ok=True)

    n_iter_map = n_iter_per_level or {0: 400, 1: 600}

    # -----------------------------------------------------------------
    # Load base mesh
    # -----------------------------------------------------------------
    xs_orig, ys_orig, bc_flag_orig, cnct_quad_orig, n_orig, n_ele_quad_orig = load_mesh()

    cnct_tri_orig = triangulate_short_diagonal(cnct_quad_orig, xs_orig, ys_orig)

    tri_pd = pv.PolyData(
        np.column_stack([xs_orig, ys_orig, np.zeros(n_orig)]),
        faces=np.hstack([np.full((len(cnct_tri_orig), 1), 3, dtype=int), cnct_tri_orig]).ravel(),
    )

    print(f"Base mesh: {n_orig} nodes, {n_ele_quad_orig} quads, {len(cnct_tri_orig)} tris")

    # -----------------------------------------------------------------
    # Mesh refinement helpers
    # -----------------------------------------------------------------
    def build_level(subdiv_level):
        if subdiv_level == 0:
            return (
                xs_orig.copy(), ys_orig.copy(),
                bc_flag_orig.copy(),
                cnct_tri_orig.copy(), cnct_quad_orig.copy(),
            )
        refined = tri_pd.subdivide(subdiv_level, subfilter="linear")
        pts_ref = np.array(refined.points)
        xs_r = pts_ref[:, 0] - pts_ref[:, 0].min()
        ys_r = pts_ref[:, 1] - pts_ref[:, 1].min()
        cnct_tri_r = refined.regular_faces
        n_node_r = refined.n_points

        boundary = refined.extract_feature_edges(
            boundary_edges=True, non_manifold_edges=False,
            feature_edges=False, manifold_edges=False
        )
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_ref)
        dists, idxs = tree.query(np.array(boundary.points))
        boundary_node_ids = set(idxs[dists < 1e-10])
        bc_flag_r = np.zeros(n_node_r, dtype=int)
        for nd in boundary_node_ids:
            bc_flag_r[nd] = 1

        cnct_quad_r = None
        if subdiv_level == 1:
            xs_q, ys_q, cnct_quad_r, _ = subdivide_quads(cnct_quad_orig, xs_orig, ys_orig)

        return xs_r, ys_r, bc_flag_r, cnct_tri_r, cnct_quad_r

    # -----------------------------------------------------------------
    # Shape optimization driver (local to convergence)
    # -----------------------------------------------------------------
    def run_opt(label, xs, ys, bc_flag, cnct, elem_type, B_ij, n_iter=400):
        n_node = len(xs)
        design_nodes = np.where(bc_flag == 0)[0]
        n_free = len(design_nodes)
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
        opt.set_step_size(step_size)
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

    # -----------------------------------------------------------------
    # Run convergence study
    # -----------------------------------------------------------------
    results = {}
    se_final = {}

    for level in levels:
        print(f"\n--- Refinement level {level} ---")
        xs_l, ys_l, bc_l, cnct_tri_l, _ = build_level(level)
        B_ij = make_filter(xs_l, ys_l)
        n_iter = n_iter_map.get(level, 400)

        # Tri shell
        key_tri = f"tri_L{level}"
        zs_opt, u_nodal, hist = run_opt(key_tri, xs_l, ys_l, bc_l, cnct_tri_l, "tri", B_ij, n_iter)
        results[key_tri] = (xs_l, ys_l, bc_l, cnct_tri_l, zs_opt, u_nodal, hist)
        se_final[key_tri] = hist[-1]

        # Beam on tri edges
        edges_l = extract_edges(cnct_tri_l)
        key_beam = f"beam_L{level}"
        zs_opt, u_nodal, hist = run_opt(key_beam, xs_l, ys_l, bc_l, edges_l, "beam", B_ij, n_iter)
        results[key_beam] = (xs_l, ys_l, bc_l, edges_l, zs_opt, u_nodal, hist)
        se_final[key_beam] = hist[-1]

        # Quad shell
        if level == 0:
            key_quad = "quad_L0"
            zs_opt, u_nodal, hist = run_opt(key_quad, xs_l, ys_l, bc_l, cnct_quad_orig, "quad", B_ij, n_iter)
            results[key_quad] = (xs_l, ys_l, bc_l, cnct_quad_orig, zs_opt, u_nodal, hist)
            se_final[key_quad] = hist[-1]
        elif level == 1:
            xs_q, ys_q, cnct_quad_L1, _ = subdivide_quads(cnct_quad_orig, xs_orig, ys_orig)
            bc_q = detect_boundary_nodes(xs_q, ys_q, cnct_quad_L1, "quad")
            B_ij_q = make_filter(xs_q, ys_q)
            key_quad = "quad_L1"
            zs_opt, u_nodal, hist = run_opt(key_quad, xs_q, ys_q, bc_q, cnct_quad_L1, "quad", B_ij_q, n_iter)
            results[key_quad] = (xs_q, ys_q, bc_q, cnct_quad_L1, zs_opt, u_nodal, hist)
            se_final[key_quad] = hist[-1]

    # -----------------------------------------------------------------
    # Export VTU files
    # -----------------------------------------------------------------
    print("\nExporting VTU files...")
    for key, (xs_l, ys_l, bc_l, cnct_l, zs_opt, u_nodal, _) in results.items():
        pts = np.column_stack([xs_l, ys_l, zs_opt])
        n_cells = len(cnct_l)

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
        grid.save(str(out / f"{key}_opt.vtu"))

    print("  Done.")

    # Save results data
    data_to_save = {}
    for key, (xs_l, ys_l, bc_l, cnct_l, zs_opt, u_nodal, hist) in results.items():
        data_to_save[key] = {
            "xs": xs_l, "ys": ys_l, "bc": bc_l, "cnct": cnct_l,
            "zs_opt": zs_opt, "u_nodal": u_nodal, "hist": np.array(hist),
        }
    np.savez(str(out / "convergence_data.npz"), **{
        f"{key}__{field}": val
        for key, fields in data_to_save.items()
        for field, val in fields.items()
    })
    print(f"  Saved {out / 'convergence_data.npz'}")

    # -----------------------------------------------------------------
    # Convergence plot
    # -----------------------------------------------------------------
    levels_all = sorted(set(int(k.split("_L")[1]) for k in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: final strain energy vs level
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

    # Panel 2: optimization history
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
    fig.savefig(str(out / "convergence_plot.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved {out / 'convergence_plot.png'}")

    # -----------------------------------------------------------------
    # Shape comparison plots
    # -----------------------------------------------------------------
    def get_tri_cnct_for_plot(key):
        xs_l, ys_l, bc_l, cnct_l, zs_opt, _, hist = results[key]
        if "quad" in key:
            return np.vstack([cnct_l[:, [0, 1, 2]], cnct_l[:, [0, 2, 3]]])
        elif "beam" in key:
            level = key.split("_L")[1]
            tri_key = f"tri_L{level}"
            if tri_key in results:
                tri_cnct = results[tri_key][3]
            else:
                tri_cnct = cnct_tri_orig
            return tri_cnct
        else:
            return cnct_l

    level_models = {}
    for key in sorted(results.keys()):
        level = int(key.split("_L")[1])
        level_models.setdefault(level, []).append(key)

    type_order = {"quad": 0, "tri": 1, "beam": 2}
    for lvl in level_models:
        level_models[lvl].sort(key=lambda k: type_order.get(k.split("_L")[0], 9))

    all_keys_flat = []
    for lvl in sorted(level_models.keys()):
        all_keys_flat.extend(level_models[lvl])

    n_cols = len(all_keys_flat)
    fig, axes_plot = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))
    if n_cols == 1:
        axes_plot = axes_plot.reshape(2, 1)

    ref_data = {}
    for lvl in sorted(level_models.keys()):
        if f"quad_L{lvl}" in results:
            rk = f"quad_L{lvl}"
        elif f"tri_L{lvl}" in results:
            rk = f"tri_L{lvl}"
        else:
            continue
        ref_data[lvl] = (results[rk][0], results[rk][1], results[rk][4], rk)

    all_z = np.concatenate([results[k][4] for k in all_keys_flat])
    z_min, z_max = all_z.min(), all_z.max()

    for col_idx, key in enumerate(all_keys_flat):
        xs_l, ys_l, bc_l, cnct_l, zs_opt, _, hist = results[key]
        tri_cnct = get_tri_cnct_for_plot(key)
        n_el = len(cnct_l)
        level = int(key.split("_L")[1])

        # Top row: optimized shape
        ax = axes_plot[0, col_idx]
        sc = ax.tripcolor(xs_l, ys_l, tri_cnct, zs_opt, cmap="viridis",
                          shading="gouraud", vmin=z_min, vmax=z_max)
        ax.set_aspect("equal")
        ax.set_title(f"{key}\n{len(xs_l)} nodes, {n_el} elems\nSE = {hist[-1]:.1f}", fontsize=10)
        ax.axis("off")
        plt.colorbar(sc, ax=ax, label="z", shrink=0.6)

        # Bottom row: error vs reference
        ax = axes_plot[1, col_idx]
        rd = ref_data.get(level)
        ref_key = rd[3] if rd else None
        if rd is not None and key != ref_key:
            xs_ref, ys_ref, zs_ref = rd[0], rd[1], rd[2]
            if len(zs_ref) == len(zs_opt):
                err = zs_opt - zs_ref
            else:
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
    fig.savefig(str(out / "shapes_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved {out / 'shapes_comparison.png'}")

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
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


###########################################################################
#
#  3. BENCHMARK
#
###########################################################################

def run_benchmark(n_reps=5, n_iter=20, out=Path("Examples/output/benchmark")):
    """
    Performance benchmark: measure JIT time, per-iteration time, and memory
    usage for each model type (tri/quad/beam) at refinement levels 0 and 1.

    Runs multiple repetitions of (JIT warmup + timed iterations) to collect
    statistics. Produces bar charts for per-iteration time, JIT time, and
    peak memory, plus a summary table.

    Parameters
    ----------
    n_reps : int
        Number of full repetitions per configuration (each includes a fresh
        JIT compilation + n_iter timed iterations).
    n_iter : int
        Number of timed value_and_grad calls per repetition (after JIT warmup).
    """
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load and prepare meshes
    # -----------------------------------------------------------------
    xs_orig, ys_orig, bc_flag_orig, cnct_quad_orig, n_orig, _ = load_mesh()
    n_free_L0 = int((bc_flag_orig == 0).sum())

    cnct_tri_L0 = triangulate_short_diagonal(cnct_quad_orig, xs_orig, ys_orig)
    tri_pd = pv.PolyData(
        np.column_stack([xs_orig, ys_orig, np.zeros(n_orig)]),
        faces=np.hstack([np.full((len(cnct_tri_L0), 1), 3, dtype=int), cnct_tri_L0]).ravel(),
    )
    refined = tri_pd.subdivide(1, subfilter="linear")
    pts_ref = np.array(refined.points)
    xs_L1 = pts_ref[:, 0] - pts_ref[:, 0].min()
    ys_L1 = pts_ref[:, 1] - pts_ref[:, 1].min()
    cnct_tri_L1 = refined.regular_faces
    bc_L1_tri = detect_boundary_nodes(xs_L1, ys_L1, cnct_tri_L1, "tri")

    xs_qL1, ys_qL1, cnct_quad_L1, _ = subdivide_quads(cnct_quad_orig, xs_orig, ys_orig)
    bc_qL1 = detect_boundary_nodes(xs_qL1, ys_qL1, cnct_quad_L1, "quad")

    edges_L0 = extract_edges(cnct_tri_L0)
    edges_L1 = extract_edges(cnct_tri_L1)

    configs = {
        "tri_L0":  (xs_orig.copy(), ys_orig.copy(), bc_flag_orig.copy(), cnct_tri_L0, "tri"),
        "quad_L0": (xs_orig.copy(), ys_orig.copy(), bc_flag_orig.copy(), cnct_quad_orig, "quad"),
        "beam_L0": (xs_orig.copy(), ys_orig.copy(), bc_flag_orig.copy(), edges_L0, "beam"),
        "tri_L1":  (xs_L1, ys_L1, bc_L1_tri, cnct_tri_L1, "tri"),
        "quad_L1": (xs_qL1, ys_qL1, bc_qL1, cnct_quad_L1, "quad"),
        "beam_L1": (xs_L1, ys_L1, bc_L1_tri, edges_L1, "beam"),
    }

    print(f"Configurations to benchmark:")
    for key, (xs_c, ys_c, bc_c, cnct_c, etype) in configs.items():
        print(f"  {key:10s}: {len(xs_c):5d} nodes, {len(cnct_c):5d} {etype} elements")

    # -----------------------------------------------------------------
    # Benchmark helpers
    # -----------------------------------------------------------------
    def build_sso(xs, ys, bc_flag, cnct, elem_type):
        n_node = len(xs)
        design_nodes = np.where(bc_flag == 0)[0]
        n_free = len(design_nodes)
        load_per_node = load * n_free_L0 / n_free
        B_ij = make_filter(xs, ys)

        rng = np.random.RandomState(42)
        zs = np.zeros(n_node)
        zs[design_nodes] = rng.uniform(0.5, 0.51, size=n_free)
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
        return sso, design_nodes, zs, B_ij

    def benchmark_one(label, xs, ys, bc_flag, cnct, elem_type):
        n_node = len(xs)
        gpu_backend = jax.default_backend() == "gpu"
        gpu_devices = jax.local_devices()

        jit_times = []
        iter_all = []
        peak_cpu_mbs = []
        peak_gpu_mbs = []
        cpu_parallelisms = []

        for rep in range(n_reps):
            sso, design_nodes, zs, B_ij = build_sso(xs, ys, bc_flag, cnct, elem_type)

            def do_value_and_grad():
                p = zs.copy()
                p[design_nodes] = np.asarray(sso.nodeparameters_values)
                z = (B_ij @ p)[design_nodes]
                sso.update_nodeparameter(z)
                C, sens = sso.value_grad_params(which_solver="sparse", enforce_scipy_sparse=True)
                return float(C)

            tracemalloc.start()
            jax.clear_caches()

            t0 = time.perf_counter()
            cpu_t0 = time.process_time()
            _ = do_value_and_grad()
            jit_wall = time.perf_counter() - t0
            jit_cpu = time.process_time() - cpu_t0
            jit_times.append(jit_wall)
            cpu_parallelisms.append(jit_cpu / jit_wall if jit_wall > 0 else 1.0)

            for _ in range(n_iter):
                t0 = time.perf_counter()
                _ = do_value_and_grad()
                iter_all.append(time.perf_counter() - t0)

            peak_cpu_mbs.append(tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()

            if gpu_backend:
                try:
                    stats = gpu_devices[0].memory_stats()
                    peak_gpu_mbs.append(stats.get("peak_bytes_in_use", 0) / 1e6)
                except Exception:
                    peak_gpu_mbs.append(0)

        jit_times = np.array(jit_times)
        iter_all = np.array(iter_all)
        peak_cpu_mbs = np.array(peak_cpu_mbs)
        peak_gpu_mbs = np.array(peak_gpu_mbs) if peak_gpu_mbs else np.array([0.0])
        cpu_parallelisms = np.array(cpu_parallelisms)

        result = {
            "label": label,
            "n_nodes": n_node,
            "n_elems": len(cnct),
            "elem_type": elem_type,
            "jit_mean_s": jit_times.mean(),
            "jit_std_s": jit_times.std(),
            "iter_mean_s": iter_all.mean(),
            "iter_std_s": iter_all.std(),
            "peak_cpu_mb_mean": peak_cpu_mbs.mean(),
            "peak_cpu_mb_std": peak_cpu_mbs.std(),
            "peak_gpu_mb_mean": peak_gpu_mbs.mean(),
            "peak_gpu_mb_std": peak_gpu_mbs.std(),
            "cpu_par_mean": cpu_parallelisms.mean(),
            "jit_times": jit_times,
            "iter_times": iter_all,
        }

        print(f"  [{label:10s}]  JIT={jit_times.mean():6.2f}+/-{jit_times.std():.2f}s  "
              f"iter={iter_all.mean():.3f}+/-{iter_all.std():.3f}s  "
              f"CPU={peak_cpu_mbs.mean():.0f}MB  GPU={peak_gpu_mbs.mean():.0f}MB")

        return result

    # -----------------------------------------------------------------
    # Run benchmarks
    # -----------------------------------------------------------------
    print(f"\nBenchmarking: {n_reps} reps x (1 JIT + {n_iter} iters) per config\n")

    bench_results = {}
    for key, (xs_c, ys_c, bc_c, cnct_c, etype) in configs.items():
        bench_results[key] = benchmark_one(key, xs_c, ys_c, bc_c, cnct_c, etype)

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
    print("\n" + "=" * 110)
    print(f"BENCHMARK SUMMARY  ({n_reps} reps x {n_iter} iters)")
    print("=" * 110)
    print(f"{'Model':<10s} {'Nodes':>6s} {'Elems':>6s} {'JIT (s)':>12s} {'Iter (s)':>14s} "
          f"{'CPU MB':>10s} {'GPU MB':>10s} {'CPU par':>8s}")
    print("-" * 110)
    for key in sorted(bench_results.keys()):
        r = bench_results[key]
        print(f"  {r['label']:<8s} {r['n_nodes']:6d} {r['n_elems']:6d} "
              f"{r['jit_mean_s']:5.1f}+/-{r['jit_std_s']:.1f}  "
              f"{r['iter_mean_s']:.3f}+/-{r['iter_std_s']:.3f}  "
              f"{r['peak_cpu_mb_mean']:5.0f}+/-{r['peak_cpu_mb_std']:.0f}  "
              f"{r['peak_gpu_mb_mean']:5.0f}+/-{r['peak_gpu_mb_std']:.0f}  "
              f"{r['cpu_par_mean']:6.1f}x")

    # -----------------------------------------------------------------
    # Save data
    # -----------------------------------------------------------------
    save_dict = {}
    for k, r in bench_results.items():
        for field, val in r.items():
            if isinstance(val, (np.ndarray, int, float)):
                save_dict[f"{k}__{field}"] = val
    np.savez(str(out / "benchmark_data.npz"), **save_dict)
    with open(str(out / "benchmark_labels.txt"), "w") as f:
        for k, r in bench_results.items():
            f.write(f"{k},{r['elem_type']},{r['n_nodes']},{r['n_elems']}\n")
    print(f"\nSaved {out / 'benchmark_data.npz'}")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    levels = sorted(set(int(k.split("_L")[1]) for k in bench_results))
    model_types = ["tri", "quad", "beam"]
    colors = {"tri": "C2", "quad": "C0", "beam": "C1"}
    labels_map = {"tri": "Tri (MITC3)", "quad": "Quad (MITC4)", "beam": "Beam (homog.)"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    bar_width = 0.25
    x_pos = np.arange(len(levels))

    def get_val(field, mtype, level, default=0):
        key = f"{mtype}_L{level}"
        if key in bench_results:
            return bench_results[key].get(field, default)
        return default

    # Panel 1: Per-iteration time
    ax = axes[0, 0]
    for i, mt in enumerate(model_types):
        vals = [get_val("iter_mean_s", mt, l) for l in levels]
        errs = [get_val("iter_std_s", mt, l) for l in levels]
        ax.bar(x_pos + i * bar_width, vals, bar_width, yerr=errs,
               color=colors[mt], label=labels_map[mt], capsize=3)
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-iteration time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: JIT compile time
    ax = axes[0, 1]
    for i, mt in enumerate(model_types):
        vals = [get_val("jit_mean_s", mt, l) for l in levels]
        errs = [get_val("jit_std_s", mt, l) for l in levels]
        ax.bar(x_pos + i * bar_width, vals, bar_width, yerr=errs,
               color=colors[mt], label=labels_map[mt], capsize=3)
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_ylabel("Time (s)")
    ax.set_title("JIT compilation time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Peak GPU memory
    ax = axes[1, 0]
    for i, mt in enumerate(model_types):
        vals = [get_val("peak_gpu_mb_mean", mt, l) for l in levels]
        errs = [get_val("peak_gpu_mb_std", mt, l) for l in levels]
        ax.bar(x_pos + i * bar_width, vals, bar_width, yerr=errs,
               color=colors[mt], label=labels_map[mt], capsize=3)
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Peak GPU memory")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Peak CPU memory
    ax = axes[1, 1]
    for i, mt in enumerate(model_types):
        vals = [get_val("peak_cpu_mb_mean", mt, l) for l in levels]
        errs = [get_val("peak_cpu_mb_std", mt, l) for l in levels]
        ax.bar(x_pos + i * bar_width, vals, bar_width, yerr=errs,
               color=colors[mt], label=labels_map[mt], capsize=3)
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Peak CPU memory")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Performance benchmark ({n_reps} reps x {n_iter} iters)", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(out / "benchmark_plot.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out / 'benchmark_plot.png'}")


###########################################################################
#
#  Main entry point
#
###########################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("  COMPARISON STUDY")
    print("=" * 70)
    run_comparison()

    print("\n\n")
    print("=" * 70)
    print("  CONVERGENCE STUDY")
    print("=" * 70)
    run_convergence()

    print("\n\n")
    print("=" * 70)
    print("  BENCHMARK")
    print("=" * 70)
    run_benchmark()
