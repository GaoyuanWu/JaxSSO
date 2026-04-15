"""
One-shot conversion: Mannheim Multihalle CSV mesh data -> VTU.

Saves a quad mesh with point field `bc_node` (1 = boundary, 0 = design).
Run once; the comparison script loads from the VTU.
"""
import numpy as np
import pyvista as pv
from pathlib import Path

data_path = Path("Examples/Data/Mannheim_Quad")
out_path = Path("Examples/Data/mannheim_quad.vtu")

# Load raw CSV data
cnct_raw = np.loadtxt(data_path / "cnct.csv", dtype=int)
n_ele = (cnct_raw.shape[0] + 1) // 4
cnct = cnct_raw.reshape(n_ele, 4)

xs = np.loadtxt(data_path / "crd_x.csv")
ys = np.loadtxt(data_path / "crd_y.csv")
zs = np.loadtxt(data_path / "crd_z.csv")
bc_nodes = np.loadtxt(data_path / "bc_node.csv", dtype=int)

n_node = xs.shape[0]

# Build PyVista quad mesh (original coordinates, not normalized)
points = np.column_stack([xs, ys, zs])
cells = np.hstack([np.full((n_ele, 1), 4, dtype=int), cnct])
celltypes = np.full(n_ele, pv.CellType.QUAD)
mesh = pv.UnstructuredGrid(cells, celltypes, points)

# Mark boundary nodes
bc_flag = np.zeros(n_node, dtype=int)
bc_flag[bc_nodes] = 1
mesh.point_data["bc_node"] = bc_flag

mesh.save(str(out_path))
print(f"Saved {out_path}  ({n_node} nodes, {n_ele} quads, {bc_nodes.shape[0]} BC nodes)")
