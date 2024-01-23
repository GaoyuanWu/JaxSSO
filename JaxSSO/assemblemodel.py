'''
This modules take 'model' object as input and assemble the linear system Ax=b(Ku=f) for the solver to solve.
'''
#%%
import numpy as np
import jax.numpy as jnp
from .beamcol import BeamCol
from .quad import Quad
from jax import vmap,jit
from jax.experimental import sparse
import jax
jax.config.update("jax_enable_x64", True)
from functools import partial
#%%
#---------------------------------------------------------------------
# Helper functions: stiffness matrix
#---------------------------------------------------------------------

@partial(jit,static_argnums=(2,3,4))
def K_aug(known_dofs,K_BCOO,ndof,ncons,k_nse):
    '''
    To impose **Boundary Conditions**, we use augmented stffness matrix using Lagrangian Multiplier method as decribed here:
    1. https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femI/Presentation.pdf
    2. https://github.com/tianjuxue/jax-am/blob/main/jax_am/fem/solver.py

    This is the LHS: augmented-stiffness matrix with boundary conditions applied.
    A sparse matrix in BCOO format.

    Parameters
    ----------

    known_dofs: 1darray
        indices of known_dofs, i.e., dofs with imposed b.c.
    
    K_BCOO: jax.experimental.sparse.BCOO
        stiffness matrix in BCOO format
    
    ndof: int
        number of dofs in the system
        
    
    k_nse: int
        number of non-zero entries of the original stiffness matrix.
            beamcol: 12*12*n_beamcol
            quad: 24*24*n_quad
            ....

    Returns
    -------
    K_aug_BCOO:jax.experimental.sparse.BCOO

    '''
    data = jnp.array([0.])
    indices = jnp.array([[0,0]],dtype='int32') 
    zero_BCOO = sparse.BCOO((data,indices),shape=(ncons,ncons)) #zero entries


    data_V = jnp.repeat(1.,ncons)
    rows = jnp.linspace(0,ncons-1,ncons,dtype='int32')
    indices_V = jnp.vstack([rows.T,known_dofs.T]).T 
    V_bcoo = sparse.BCOO((data_V,indices_V),shape=(ncons,ndof)) #zero entries

    #V_bcoo.indices = V_bcoo.indices.astype(K_BCOO.indices.dtype) #Convert V_bcoo's indices' dtype to whatever K_BCOO is using. Otherwise there will be an error.
    K_aug_row_1 = sparse.bcoo_concatenate((K_BCOO,V_bcoo.T),dimension=1)
    K_aug_row_2 = sparse.bcoo_concatenate((V_bcoo,zero_BCOO),dimension=1)
    K_aug_BCOO = sparse.bcoo_concatenate((K_aug_row_1,K_aug_row_2),dimension=0)
    
    K_aug_BCOO = K_aug_BCOO.sort_indices() #Sort indices
    nse = k_nse + 2*ncons + 1 #nse of the augmented stiffness matrix
    K_aug_BCOO_sum_dup = sparse.bcoo_sum_duplicates(K_aug_BCOO,nse) #Get rid of the duplicates (we specify nse to activate the jax functionalities.
    return K_aug_BCOO_sum_dup


@partial(jit,static_argnums=(1))
def f_aug(loads,ncons):
    '''
    To impose **Boundary Conditions**, we use augmented stffness matrix using Lagrangian Multiplier method as decribed here:
    1. https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femI/Presentation.pdf
    2. https://github.com/tianjuxue/jax-am/blob/main/jax_am/fem/solver.py

    This is the RHS: augmented-loading vector.
    A dense vector.

    Parameters
    ----------

    known_dofs: 1darray
        indices of known_dofs, i.e., dofs with imposed b.c.
    
    loads: 1darray of shape (n_dof,)
        loading vector

    
    Returns
    -------
    f_aug_dense: jnp.array
        augmented RHS
    '''
    #Load vector @ unknown indices
    f_aug_dense = jnp.concatenate((loads,np.zeros(ncons))) #augmented loading vector

    return f_aug_dense

def K_func(node_crds,ndof,n_beamcol,cnct_beamcols,prop_beamcols,n_quad,cnct_quads,prop_quads):
    '''
    Take all the parameters and return the stiffness matrix (not augmented)
    '''

    #Initial stiffness in bcoo format
    data = jnp.array([0.])
    indices = jnp.array([[0,0]],dtype='int32')
    K = sparse.BCOO((data,indices),shape=(ndof,ndof))

    #Go over different element types
    if n_beamcol>0:
        K = K + BeamCol.K_beamcol(node_crds,prop_beamcols,cnct_beamcols,ndof)
    if n_quad>0:
        K = K + Quad.K_quad(node_crds,prop_quads,cnct_quads,ndof)
    
    return K

def K_aug_func(node_crds,ndof,known_dofs,n_beamcol,cnct_beamcols,prop_beamcols,n_quad,cnct_quads,prop_quads):
    '''
    Take all the parameters and return the (augmented) stiffness matrix
    '''

    #Preparation
    ncons = known_dofs.shape[0]#dofs of the system
    k_nse = n_beamcol*12*12 #Beamcols
    k_nse += n_quad*24*24 #Quads
    k_nse += 1 #initialization entry in model_K

    #Stiffness matrix
    K_bcoo = K_func(node_crds,ndof,n_beamcol,cnct_beamcols,prop_beamcols,n_quad,cnct_quads,prop_quads) #original
    K_aug_res = K_aug(known_dofs,K_bcoo,ndof,ncons,k_nse) #augmented
    return K_aug_res

def f_aug_func(loads,known_dofs):
    '''
    Take all the parameters and return the (augmented) loading vector
    '''
    ncons = known_dofs.shape[0]
    return f_aug(loads,ncons)


#---------------------------------------------------------------------
# Helper for 'model' objects
#---------------------------------------------------------------------
        
def model_K(model):
    '''
    Given a model, return the global stiffness matrix (not augmented) in BCOO format.
    
    Parameters:
    ----------
    model:JAX_FEA's Model object

    Returns:
    K: jax.experimental.sparse.BCOO
        Global stiffness matrix w/o applying boundary conditions
    '''
    K = K_func(model.crds,model.ndof,model.n_beamcol,model.cnct_beamcols,model.prop_beamcols,
            model.n_quad,model.cnct_quads,model.prop_quads)

    return K 


def model_K_aug(model):
    '''
    Given a model, return the global stiffness matrix (augmented) in BCOO format.
    
    Parameters:
    ----------
    model:JAX_FEA's Model object

    Returns:
    Augmented global stiffness matrix with bc in jax.experimental.sparse.BCOO
    '''
    K_aug = K_aug_func(model.crds,model.ndof,model.known_id,model.n_beamcol,model.cnct_beamcols,model.prop_beamcols,
            model.n_quad,model.cnct_quads,model.prop_quads)

    return K_aug


def model_f_aug(model):
    '''
    Given a model, return the global stiffness matrix (augmented) in BCOO format.
    
    Parameters:
    ----------
    model:JAX_FEA's Model object

    Returns:
    Augmented loading vector (RHS) in dense matrix
    '''
    return f_aug_func(model.nodal_loads,model.known_id)

