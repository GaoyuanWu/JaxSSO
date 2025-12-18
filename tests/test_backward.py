#%%
import fea_problem
import FD_problem
import numpy as np
import jax
import pytest
from JaxSSO.SSO_model import NodeParameter,SSO_model

#Some jax configurations
jax.config.update("jax_enable_x64", True) #Enable jax 64-bit mode
# NOTE:
# Not enforcing GPU here in tests; let JAX decide based on environment.
# jax.config.update("jax_platform_name", "gpu")

#Solvers:dense or sparse
@pytest.mark.parametrize(
    "which_solver",
    [
        "dense",
        "sparse",
    ],
)
def test_model_ad(which_solver):
    '''
    Test whether JaxSSO ontains the correct gradient.
        Loss: total starin energy of the system
        Design variables: nodal movements along z-direction

    Parameters:
        which_solver: dense or sparse

    Benchmark from finite difference with delta_param of 10^-5

    '''
    # ------------------------------------------------------------------
    # From FEA model, create SSO model (for backward propagation)
    # ------------------------------------------------------------------
    model = fea_problem.fea_problem_model() #FE model
    sso_model = SSO_model(model) # initial sso model
    n_node = len(model.nodes) # How many nodes in the model
    design_i = int(n_node/2)-1 #Mid-node
    design_nodes = np.array([i for i in range(n_node) if i!=0 and i!=n_node-1]) #Nodes that are design nodes
    for node in design_nodes:
        nodeparameter = NodeParameter(node,2) # nodeparamter object, z-coordinate of each node
        sso_model.add_nodeparameter(nodeparameter)

    #Initial the parameters
    sso_model.initialize_parameters_values()
    sso_model.set_objective(objective='strain energy',func=None,func_args=None) #set loss to be strain energy

    # ------------------------------------------------------------------
    # Solve using JaxSSO solver
    # ------------------------------------------------------------------
    model.solve(which_solver=which_solver) #forward
    SSO_grad = float(sso_model.value_grad_params(which_solver=which_solver,enforce_scipy_sparse = True)[1][design_i]) #backward


    # ------------------------------------------------------------------
    # Post-processing: strain energy = 0.5*f.T@u
    # ------------------------------------------------------------------
    u = model.u #Solution to [K]{u}={f}
    f = model.get_loads() # {f}

    strain_energy_jaxsso = 0.5 * u @ f
    strain_energy_jaxsso = float(strain_energy_jaxsso) # Convert to scalar (in case of JAX array)

    # ------------------------------------------------------------------
    # Benchmark results from finite difference
    # ------------------------------------------------------------------
    dz = 1e-5 #finite difference step
    C_temp = FD_problem.FD_problem_model_strain_energy(design_i,which_solver,dz)
    FD_grad = ((float(C_temp-strain_energy_jaxsso)/dz)) #Finite difference result

    rtol = 5e-2   # relative tolerance

    # ------------------------------------------------------------------
    # Test results
    # ------------------------------------------------------------------


    assert np.isclose(
        FD_grad,
        SSO_grad,
        rtol=rtol,
    ), (
        f"Dense solver failed:\n"
        f"Jaxsso computed = {SSO_grad},\n "
        f"Finite difference benchmark = {FD_grad},\n"
    )

    print("Test passed.")

