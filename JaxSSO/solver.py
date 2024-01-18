'''
This modules take 'model' object as input and solve the linear system Ax=b ([K]{u}={f}).



            
There are multiple choices for solving linear system Ax=b from Jax.

*Dense solver:
    1. jnp.linalg.solve (fully developed)
*Sparse solver:
    1. jax.experimental.sparse.linalg.spsolve
        It is sparse-direct solver, not iterative.
        However, it is still under developement so it does not support all functionalities (such as AD)
    2.  jax.scipy.sparse.linalg.bicgstab/cg/gmres
        It is sparse-iterative solver. Performance needs tuning parameters.
        It is fully developed by Jax.
        Used by Jax-FEM: https://github.com/tianjuxue/jax-am/blob/main/jax_am/fem/solver.py

*Disscussion on the CUDA GPU backend
CPU sparse solve may be faster than GPU. From Jax-FDM:https://github.com/arpastrana/jax_fdm/blob/main/src/jax_fdm/equilibrium/sparse.py

'''
import numpy as np
 

import jax.numpy as jnp
from jax import vmap,jit
from jax.experimental import sparse
import jax
jax.config.update("jax_enable_x64", True)

from functools import partial

from scipy.sparse.linalg import spsolve as spsolve_scipy
from scipy.sparse import csr_matrix as scipy_csr_matrix #csr_matrix helper

#------------------------------------------
# Helper functions
#------------------------------------------
@partial(jit,static_argnums=(2,))
def convert_BCOO_to_CSR(bcoo_matrix,nse,ndof_aug):
    '''
    Convert matrix in BCOO format to jax.experimental.sparse.CSR,
    with the help of vmap.
    '''

    data = bcoo_matrix.data
    row_indices = bcoo_matrix.indices[:,0]
    col_indices = bcoo_matrix.indices[:,1]


    coo_sorted_indices = jnp.argsort(row_indices,kind='stable') # sorted indices of arrays, using rows as references
    row_indices_sorted = jnp.take(row_indices,coo_sorted_indices) #sorted rows
    col_indices_sorted = jnp.take(col_indices,coo_sorted_indices) #sorted cols
    data_sorted = jnp.take(data,coo_sorted_indices) #sorted datas

    def find_indptr_i(arr,i):
        return jnp.nonzero(arr==i,size=1)[0][0]
    

    finding_arr = jnp.linspace(0,ndof_aug-1,ndof_aug,dtype='int32') #find the first occurence of [0,1,...,ndof-1] 
    indptr_incom = vmap(find_indptr_i,in_axes=(None,0))(row_indices_sorted,finding_arr)
    indptr = jnp.append(indptr_incom,nse)
    indptr = indptr.astype('int32')
    col_indices_sorted = col_indices_sorted.astype('int32')

    return sparse.CSR((data_sorted,col_indices_sorted,indptr),
               shape=(ndof_aug,ndof_aug))

#------------------------------------------
# Solvers: dense & direct
#------------------------------------------

def jax_dense_solve(K_aug,f_aug):
    '''
    Direct dense solving of Ax=b using jnp.linalg.solve

    Parameters:
    K_aug:
        jax.experimental.BCOO format of the augmented stiffness matrix
    
    f_aug:
        ndarray, RHS of the (augmented) linear system

    Returns:
    u:
        1darray of the discplacement vector corresponding to the dofs of the system
    '''
    #Solving FEA, entended dimension

    
    return jnp.linalg.solve(K_aug.todense(), f_aug)

#------------------------------------------
# Solvers: sparse & direct from jax, GPU friendly
#------------------------------------------

# Customized vjp coz using the adjoint method, coz jax doesn't yet supported AD for 
# jax.experiment.sparse.linalg.spsolve

@jax.custom_vjp
def jax_sparse_solve(K_aug, f_aug):
    '''

    Direct sparse solving of Ax=b using the experimental module of jax.
    jax.experimental.sparse.linalg.spsolve.
    Support GPU.
    Need to define custom 'vjp' to enable the auto-diff functionality.

    Parameters:
    K_aug:
        jax.experimental.BCOO format of the augmented stiffness matrix
    
    f_aug:
        ndarray, RHS of the (augmented) linear system

        
    Returns:
    u:
        1darray of the discplacement vector corresponding to the dofs of the system, augmented.
    '''
    K_aug_bcsr = sparse.BCSR.from_bcoo(K_aug) #Convert to BCSR
    return sparse.linalg.spsolve(K_aug_bcsr.data,K_aug_bcsr.indices,K_aug_bcsr.indptr,f_aug)


def jax_sparse_solve_fwd(K_aug, f_aug):
    """
    Forward pass of jax_sparse_solve.
    """
    u_aug = jax_sparse_solve(K_aug, f_aug)


    return u_aug, (u_aug, K_aug, f_aug)


def jax_sparse_solve_bwd(res, g):
    """
    Backward pass of jax_sparse_solve.

    "g" is the partial derivative with respect to "u_aug", i.e., the primal output of the function "jax_sparse_solve"
    "lambda" is the langrangian multiplier, also the solution to the adjoint equation. We solve the adjoint eq. for lambda.
    Linear system: f(K,f)= Ku-f=0
    vjp= -lambda.T * partial_f/partial_params, where params are (K,f)
    To calculate this vjp, simply call jax.vjp for f(K,f) @ primal = (K,f) and tangent = lambda
    
    """
    u_aug, K_aug, f_aug = res

    # Solve adjoint system for lambda
    # A.T @ lambda = g
    lam = jax_sparse_solve(K_aug.T, g)
    #lam = jax_dense_solve(K_aug.T, g) #see if the "batch for spsolve" error dissappears. It did, but raised other errors (pytree structure not match of input and output)

    # the implicit constraint function for implicit differentiation
    def f_Ax_b(params):
        K_aug, f_aug = params
        return -1*(K_aug@u_aug - f_aug)

    params = (K_aug, f_aug)

    # Call vjp of residual_fn to compute gradient wrt params
    f_primal,vjp_f = jax.vjp(f_Ax_b, params) #Get the function vjp for f_Ax_b @ (K_aug, f_aug)

    return vjp_f(lam)[0] #Call it @ lamdba, index [0] is nothing but matching the pytree structure, as vjp_f(lam) returns (res,)

jax_sparse_solve.defvjp(jax_sparse_solve_fwd, jax_sparse_solve_bwd) #Register vjp for backward mode AD

#------------------------------------------
# Solvers: sparse & direct from scipy
#------------------------------------------

# Customized vjp coz using the adjoint method, coz scipy doesn't supported AD

@jax.custom_vjp
def sci_sparse_solve(K_aug, f_aug):
    '''
    Direct sparse solving of Ax=b using the experimental module of pure scipy.
    scipy.sparse.linalg.spsolve.
    Support only CPU.
    Need to define custom 'vjp' to enable the auto-diff functionality.

    Parameters:
    K_aug:
        jax.experimental.BCOO format of the augmented stiffness matrix
    
    f_aug:
        ndarray, RHS of the (augmented) linear system
        
    Returns:
    u:
        1darray of the discplacement vector corresponding to the dofs of the system, augmented.
    '''
    def callback(data, indices, indptr, _b):
        _A = scipy_csr_matrix((data, indices, indptr))
        return spsolve_scipy(_A, _b)

    K_aug_bcsr = sparse.BCSR.from_bcoo(K_aug) #Convert to BCSR
    u = jax.pure_callback(callback,  # callback function
                           f_aug,  # return type
                           np.array(K_aug_bcsr.data), 
                           np.array(K_aug_bcsr.indices,dtype='int32'),
                           np.array(K_aug_bcsr.indptr,dtype='int32'),
                           f_aug)
    return u

def sci_sparse_solve_fwd(K_aug, f_aug):
    """
    Forward pass of sci_sparse_solve.
    """
    u_aug = sci_sparse_solve(K_aug, f_aug)

    return u_aug, (u_aug, K_aug, f_aug)


def sci_sparse_solve_bwd(res, g):
    """
    Backward pass of jax_sparse_solve.

    "g" is the partial derivative with respect to "u_aug", i.e., the primal output of the function "jax_sparse_solve"
    "lambda" is the langrangian multiplier, also the solution to the adjoint equation. We solve the adjoint eq. for lambda.
    Linear system: f(K,f)= Ku-f=0
    vjp= -lambda.T * partial_f/partial_params, where params are (K,f)
    To calculate this vjp, simply call jax.vjp for f(K,f) @ primal = (K,f) and tangent = lambda
    
    """
    u_aug, K_aug, f_aug = res

    # Solve adjoint system for lambda
    # A.T @ lambda = g
    lam = sci_sparse_solve(K_aug.T, g)

    # the implicit constraint function for implicit differentiation
    def f_Ax_b(params):
        K_aug, f_aug = params
        return K_aug@u_aug - f_aug

    params = (K_aug, f_aug)

    # Call vjp of residual_fn to compute gradient wrt params
    vjp_f = jax.vjp(f_Ax_b, params)[1] #Get the function vjp for f_Ax_b @ (K_aug, f_aug)

    return -1*vjp_f(lam) #Call it @ lamdba

sci_sparse_solve.defvjp(sci_sparse_solve_fwd, sci_sparse_solve_bwd) #Register vjp for backward mode AD


