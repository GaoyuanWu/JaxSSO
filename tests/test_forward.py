#%%
import fea_problem
import numpy as np
import jax
import pytest

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
def test_model_solve(which_solver):
    '''
    Test whether the JaxSSO solver obtains correct solutions.

    Parameters:
        which_solver: dense or sparse

    Benchmark from external solver (SAP2000):
        strain_energy = 3503.7935 

    '''
    # ------------------------------------------------------------------
    # Create FEA model
    # ------------------------------------------------------------------
    model = fea_problem.fea_problem_model()

    # ------------------------------------------------------------------
    # Benchmark results from commercial solver (SAP2000)
    # ------------------------------------------------------------------
    strain_energy_bench = 3503.7935
    rtol = 1e-3   # relative tolerance

    # ------------------------------------------------------------------
    # Solve using JaxSSO dense solver
    # ------------------------------------------------------------------
    model.solve(which_solver=which_solver)

    # ------------------------------------------------------------------
    # Post-processing: strain energy = 0.5*f.T@u
    # ------------------------------------------------------------------
    u = model.u #Solution to [K]{u}={f}
    f = model.get_loads() # {f}

    strain_energy_jaxsso = 0.5 * u @ f

    # Convert to scalar (in case of JAX array)
    strain_energy_jaxsso = float(strain_energy_jaxsso)

    # ------------------------------------------------------------------
    # Test results
    # ------------------------------------------------------------------


    assert np.isclose(
        strain_energy_jaxsso,
        strain_energy_bench,
        rtol=rtol
    ), (
        f"Dense solver failed:\n"
        f"Jaxsso computed = {strain_energy_jaxsso},\n "
        f"benchmark = {strain_energy_bench},\n"
    )

    print("Test passed.")

