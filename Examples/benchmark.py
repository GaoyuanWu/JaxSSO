"""
Performance benchmark: measure JIT time, per-iteration time, and memory usage
for each model type (tri/quad/beam) at each refinement level.

Runs a small number of optimization iterations (not a full convergence study)
to isolate FEA/AD cost from optimizer behavior.
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

OUT = Path("Examples/output/benchmark")
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================================
# 1. Load and prepare meshes (reuse logic from convergence_study.py)
# =========================================================================
mesh = pv.read("Examples/Data/mannheim_quad.vtu")
raw_pts = np.array(mesh.points)
bc_flag_orig = np.array(mesh.point_data["bc_node"])
n_orig = mesh.n_points
cnct_quad_orig = mesh.cells.reshape(-1, 5)[:, 1:]
xs_orig = raw_pts[:, 0] - raw_pts[:, 0].min()
ys_orig = raw_pts[:, 1] - raw_pts[:, 1].min()

# Material
t = 0.1
E = 1e10
nu = 0.3
load = 5000.0
n_free_L0 = int((bc_flag_orig == 0).sum())


def triangulate_short_diagonal(cnct_quad, xs, ys):
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
    return np.array(xs_new), np.array(ys_new), np.array(cnct_new, dtype=int)


def extract_edges(cnct_tri):
    edge_set = set()
    for f in cnct_tri:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            edge_set.add((min(a, b), max(a, b)))
    return np.array(sorted(edge_set), dtype=int)


def detect_boundary_nodes(xs, ys, cnct, cell_type="tri"):
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


def beam_props_tri_homog(E, t, d):
    s3h = np.sqrt(3) / 2
    A = s3h * t * d
    Iy = s3h * t**3 * d / 12
    Iz = Iy
    J = Iy + Iz
    G_b = E / (2 * (1 + 1 / 3))
    return A, Iy, Iz, J, G_b


def make_filter(xs, ys, R=10.0):
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D = np.sqrt(dx**2 + dy**2)
    B_raw = np.where(D > R, 0.0, (1.0 / R) * (R - D))
    return B_raw / B_raw.sum(axis=1, keepdims=True)


# =========================================================================
# 2. Build all mesh configurations
# =========================================================================
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

xs_qL1, ys_qL1, cnct_quad_L1 = subdivide_quads(cnct_quad_orig, xs_orig, ys_orig)
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
for key, (xs, ys, bc, cnct, etype) in configs.items():
    print(f"  {key:10s}: {len(xs):5d} nodes, {len(cnct):5d} {etype} elements")


# =========================================================================
# 3. Benchmark function
# =========================================================================
N_REPS = 5     # full repetitions for statistics
N_ITER = 20    # timed iterations per rep (after JIT warmup)


def build_sso(xs, ys, bc_flag, cnct, elem_type):
    """Build a model and SSO, return (sso, design_nodes, zs, B_ij)."""
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


def benchmark(label, xs, ys, bc_flag, cnct, elem_type):
    """
    Run N_REPS repetitions of (JIT + N_ITER iterations).
    Returns aggregated statistics.
    """
    n_node = len(xs)
    gpu_backend = jax.default_backend() == "gpu"
    gpu_devices = jax.local_devices()

    jit_times = []
    iter_all = []  # all individual iteration times across reps
    peak_cpu_mbs = []
    peak_gpu_mbs = []
    cpu_parallelisms = []

    for rep in range(N_REPS):
        sso, design_nodes, zs, B_ij = build_sso(xs, ys, bc_flag, cnct, elem_type)

        def do_value_and_grad():
            p = zs.copy()
            p[design_nodes] = np.asarray(sso.nodeparameters_values)
            z = (B_ij @ p)[design_nodes]
            sso.update_nodeparameter(z)
            C, sens = sso.value_grad_params(which_solver="sparse", enforce_scipy_sparse=True)
            return float(C)

        # JIT (first call with fresh caches)
        tracemalloc.start()
        jax.clear_caches()

        t0 = time.perf_counter()
        cpu_t0 = time.process_time()
        _ = do_value_and_grad()
        jit_wall = time.perf_counter() - t0
        jit_cpu = time.process_time() - cpu_t0
        jit_times.append(jit_wall)
        cpu_parallelisms.append(jit_cpu / jit_wall if jit_wall > 0 else 1.0)

        # Timed iterations
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            _ = do_value_and_grad()
            iter_all.append(time.perf_counter() - t0)

        # Memory
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
        # Store raw arrays for detailed plotting
        "jit_times": jit_times,
        "iter_times": iter_all,
    }

    print(f"  [{label:10s}]  JIT={jit_times.mean():6.2f}+/-{jit_times.std():.2f}s  "
          f"iter={iter_all.mean():.3f}+/-{iter_all.std():.3f}s  "
          f"CPU={peak_cpu_mbs.mean():.0f}MB  GPU={peak_gpu_mbs.mean():.0f}MB")

    return result


# =========================================================================
# 4. Run benchmarks
# =========================================================================
print(f"\nBenchmarking: {N_REPS} reps x (1 JIT + {N_ITER} iters) per config\n")

bench_results = {}
for key, (xs, ys, bc, cnct, etype) in configs.items():
    bench_results[key] = benchmark(key, xs, ys, bc, cnct, etype)


# =========================================================================
# 5. Summary table
# =========================================================================
print("\n" + "=" * 110)
print(f"BENCHMARK SUMMARY  ({N_REPS} reps x {N_ITER} iters)")
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


# =========================================================================
# 6. Save data
# =========================================================================
save_dict = {}
for k, r in bench_results.items():
    for field, val in r.items():
        if isinstance(val, (np.ndarray, int, float)):
            save_dict[f"{k}__{field}"] = val
np.savez(str(OUT / "benchmark_data.npz"), **save_dict)
with open(str(OUT / "benchmark_labels.txt"), "w") as f:
    for k, r in bench_results.items():
        f.write(f"{k},{r['elem_type']},{r['n_nodes']},{r['n_elems']}\n")
print(f"\nSaved {OUT / 'benchmark_data.npz'}")


# =========================================================================
# 7. Plots
# =========================================================================
levels = sorted(set(int(k.split("_L")[1]) for k in bench_results))
model_types = ["tri", "quad", "beam"]
colors = {"tri": "C2", "quad": "C0", "beam": "C1"}
labels = {"tri": "Tri (MITC3)", "quad": "Quad (MITC4)", "beam": "Beam (homog.)"}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

bar_width = 0.25
x_pos = np.arange(len(levels))

# Helper to get value for a model type at a level
def get_val(field, mtype, level, default=0):
    key = f"{mtype}_L{level}"
    if key in bench_results:
        return bench_results[key].get(field, default)
    return default

# --- Panel 1: Per-iteration time ---
ax = axes[0, 0]
for i, mt in enumerate(model_types):
    vals = [get_val("iter_mean_s", mt, l) for l in levels]
    errs = [get_val("iter_std_s", mt, l) for l in levels]
    ax.bar(x_pos + i * bar_width, vals, bar_width, yerr=errs,
           color=colors[mt], label=labels[mt], capsize=3)
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels([f"L{l}" for l in levels])
ax.set_ylabel("Time (s)")
ax.set_title("Per-iteration time")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 2: JIT compile time ---
ax = axes[0, 1]
for i, mt in enumerate(model_types):
    vals = [get_val("jit_mean_s", mt, l) for l in levels]
    errs = [get_val("jit_std_s", mt, l) for l in levels]
    ax.bar(x_pos + i * bar_width, vals, bar_width, yerr=errs,
           color=colors[mt], label=labels[mt], capsize=3)
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels([f"L{l}" for l in levels])
ax.set_ylabel("Time (s)")
ax.set_title("JIT compilation time")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 3: Peak GPU memory ---
ax = axes[1, 0]
for i, mt in enumerate(model_types):
    vals = [get_val("peak_gpu_mb_mean", mt, l) for l in levels]
    errs = [get_val("peak_gpu_mb_std", mt, l) for l in levels]
    ax.bar(x_pos + i * bar_width, vals, bar_width, yerr=errs,
           color=colors[mt], label=labels[mt], capsize=3)
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels([f"L{l}" for l in levels])
ax.set_ylabel("Memory (MB)")
ax.set_title("Peak GPU memory")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 4: Peak CPU memory ---
ax = axes[1, 1]
for i, mt in enumerate(model_types):
    vals = [get_val("peak_cpu_mb_mean", mt, l) for l in levels]
    errs = [get_val("peak_cpu_mb_std", mt, l) for l in levels]
    ax.bar(x_pos + i * bar_width, vals, bar_width, yerr=errs,
           color=colors[mt], label=labels[mt], capsize=3)
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels([f"L{l}" for l in levels])
ax.set_ylabel("Memory (MB)")
ax.set_title("Peak CPU memory")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Performance benchmark ({N_REPS} reps x {N_ITER} iters)", fontsize=14)
fig.tight_layout()
fig.savefig(str(OUT / "benchmark_plot.png"), dpi=150, bbox_inches="tight")
print(f"Saved {OUT / 'benchmark_plot.png'}")
