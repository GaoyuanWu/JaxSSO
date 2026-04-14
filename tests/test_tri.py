#%%
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import JaxSSO.model as Model
from JaxSSO.element import Tri
from JaxSSO.SSO_model import NodeParameter, SSO_model

jax.config.update("jax_enable_x64", True)


# -------------------------------------------------------------------
# Test 1: Rigid body modes — 18x18 local K must have exactly 6 zero eigenvalues
# -------------------------------------------------------------------
def test_tri_rigid_body_modes():
    '''A single triangle element should have exactly 6 zero-energy rigid body modes.'''
    crds = jnp.array([0.0, 0.0, 0.0,
                       2.0, 0.0, 0.0,
                       1.0, 1.5, 0.0])
    K_local = Tri.element_K_tri(crds, t=0.1, E=1e10, nu=0.3, kx_mod=1.0, ky_mod=1.0)

    eigvals = jnp.linalg.eigvalsh(K_local)
    eigvals_sorted = jnp.sort(jnp.abs(eigvals))

    # First 6 eigenvalues should be ~0, rest positive
    n_zero = jnp.sum(eigvals_sorted < 1e-6 * eigvals_sorted[-1])
    assert int(n_zero) == 6, (
        f"Expected 6 zero eigenvalues (rigid body modes), got {int(n_zero)}.\n"
        f"Smallest eigenvalues: {eigvals_sorted[:8]}"
    )


# -------------------------------------------------------------------
# Test 2: Symmetry — K must be symmetric
# -------------------------------------------------------------------
def test_tri_symmetry():
    '''Element stiffness matrix must be symmetric.'''
    crds = jnp.array([0.0, 0.0, 0.0,
                       3.0, 0.0, 0.0,
                       1.0, 2.0, 0.5])
    K = Tri.element_K_tri(crds, t=0.1, E=1e10, nu=0.3, kx_mod=1.0, ky_mod=1.0)
    assert jnp.allclose(K, K.T, atol=1e-6), "Stiffness matrix is not symmetric"


# -------------------------------------------------------------------
# Test 3: Patch test — constant membrane strain under uniform tension
# -------------------------------------------------------------------
@pytest.mark.parametrize("which_solver", ["dense", "sparse"])
def test_tri_patch_test(which_solver):
    '''
    Patch test: a rectangular plate modeled with 2 triangles under
    uniform tension in x. All interior nodes should have the same
    stress (constant strain state).

    Geometry:
        3 --- 2
        |  /  |
        0 --- 1

    Boundary: nodes 0,3 fixed in x; all fixed in y,z and rotations.
    Load: nodes 1,2 pulled in +x with force = sigma * t * Ly / 2.
    '''
    Lx, Ly = 2.0, 1.0
    t = 0.1
    E = 1e7
    nu = 0.3
    sigma_x = 1000.0  # target stress

    model = Model.Model()
    model.add_node(0, 0.0, 0.0, 0.0)
    model.add_node(1, Lx,  0.0, 0.0)
    model.add_node(2, Lx,  Ly,  0.0)
    model.add_node(3, 0.0, Ly,  0.0)

    # Fix left edge (x), all y, z, rotations everywhere
    model.add_support(0, [1, 1, 1, 1, 1, 1])
    model.add_support(3, [1, 1, 1, 1, 1, 1])
    # Right edge: fix y, z, rotations, free in x
    model.add_support(1, [0, 1, 1, 1, 1, 1])
    model.add_support(2, [0, 1, 1, 1, 1, 1])

    # Applied force on right edge: sigma * t * tributary_length
    F_per_node = sigma_x * t * Ly / 2.0
    model.add_nodal_load(1, nodal_load=[F_per_node, 0, 0, 0, 0, 0])
    model.add_nodal_load(2, nodal_load=[F_per_node, 0, 0, 0, 0, 0])

    # Two triangles: (0,1,2) and (0,2,3)
    model.add_tri(0, 0, 1, 2, t, E, nu)
    model.add_tri(1, 0, 2, 3, t, E, nu)

    model.solve(which_solver=which_solver)
    u = model.u

    # With y constrained (eps_yy = 0), this is plane strain:
    # sigma_x = E/(1-nu^2) * eps_x, so eps_x = sigma_x*(1-nu^2)/E
    # u_x = eps_x * Lx
    u_x_expected = sigma_x * (1 - nu**2) * Lx / E
    u_x_node1 = float(u[1 * 6 + 0])
    u_x_node2 = float(u[2 * 6 + 0])

    rtol = 5e-2
    assert np.isclose(u_x_node1, u_x_expected, rtol=rtol), (
        f"Patch test failed at node 1: u_x={u_x_node1}, expected={u_x_expected}"
    )
    assert np.isclose(u_x_node2, u_x_expected, rtol=rtol), (
        f"Patch test failed at node 2: u_x={u_x_node2}, expected={u_x_expected}"
    )
    # Both right-edge nodes should have the same displacement (uniform strain)
    assert np.isclose(u_x_node1, u_x_node2, rtol=1e-10), (
        f"Non-uniform displacement: node1={u_x_node1}, node2={u_x_node2}"
    )


# -------------------------------------------------------------------
# Test 4: Gradient verification — compare AD gradient with finite difference
# -------------------------------------------------------------------
@pytest.mark.parametrize("which_solver", ["dense", "sparse"])
def test_tri_gradient(which_solver):
    '''
    Test AD gradient of strain energy w.r.t. z-coordinate of a design node,
    comparing against finite difference.

    Geometry: simply-supported arch of triangles (2 tris forming a strip).
    '''
    t = 0.1
    E = 1e8
    nu = 0.3
    Q = 100.0

    # 4-node strip: 0--1--2--3 (simply supported at 0 and 3)
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 0.0, 0.0, 0.0]
    zs = [0.0, 0.3, 0.3, 0.0]

    def build_model(z_perturb=0.0, node_idx=1):
        m = Model.Model()
        for i in range(4):
            z = zs[i] + (z_perturb if i == node_idx else 0.0)
            m.add_node(i, xs[i], ys[i], z)
        m.add_support(0, [1, 1, 1, 1, 1, 1])
        m.add_support(3, [1, 1, 1, 1, 1, 1])
        m.add_nodal_load(1, [0, 0, -Q, 0, 0, 0])
        m.add_nodal_load(2, [0, 0, -Q, 0, 0, 0])
        # Two quads split into 4 triangles
        m.add_tri(0, 0, 1, 2, t, E, nu)
        m.add_tri(1, 0, 2, 3, t, E, nu)
        return m

    # Forward solve
    model = build_model()
    model.solve(which_solver=which_solver)
    u = model.u
    f = model.get_loads()
    SE = float(0.5 * u @ f)

    # AD gradient
    sso = SSO_model(model)
    design_node = 1
    sso.add_nodeparameter(NodeParameter(design_node, 2))  # z-coordinate
    sso.initialize_parameters_values()
    sso.set_objective(objective='strain energy')
    _, grad = sso.value_grad_params(which_solver=which_solver, enforce_scipy_sparse=True)
    ad_grad = float(grad[0])

    # Finite difference
    dz = 1e-5
    model_pert = build_model(z_perturb=dz, node_idx=design_node)
    model_pert.solve(which_solver=which_solver)
    u_pert = model_pert.u
    f_pert = model_pert.get_loads()
    SE_pert = float(0.5 * u_pert @ f_pert)
    fd_grad = (SE_pert - SE) / dz

    rtol = 5e-2
    assert np.isclose(ad_grad, fd_grad, rtol=rtol), (
        f"Gradient mismatch:\n  AD = {ad_grad}\n  FD = {fd_grad}\n"
        f"  relative error = {abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-30):.4e}"
    )


# -------------------------------------------------------------------
# Test 5: Non-planar triangle — element works for 3D orientation
# -------------------------------------------------------------------
def test_tri_3d_orientation():
    '''Element stiffness should work for a triangle not in the xy-plane.'''
    crds = jnp.array([0.0, 0.0, 0.0,
                       1.0, 0.0, 1.0,
                       0.5, 1.0, 0.5])
    K = Tri.element_K_tri(crds, t=0.1, E=1e10, nu=0.3, kx_mod=1.0, ky_mod=1.0)

    # Should be symmetric
    assert jnp.allclose(K, K.T, atol=1e-10), "3D stiffness matrix is not symmetric"

    # Should have 6 zero eigenvalues
    eigvals = jnp.sort(jnp.abs(jnp.linalg.eigvalsh(K)))
    n_zero = jnp.sum(eigvals < 1e-6 * eigvals[-1])
    assert int(n_zero) == 6, f"Expected 6 zero eigenvalues in 3D, got {int(n_zero)}"
