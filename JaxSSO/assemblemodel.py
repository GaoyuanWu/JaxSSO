'''
This modules take 'model' object as input and assemble the linear system Ax=b(Ku=f) for the solver to solve.
'''
#%%
import numpy as np
import jax.numpy as jnp
from .element import BeamCol,Quad
from jax import vmap,jit,custom_jvp,jacfwd
from jax.experimental import sparse
import jax
jax.config.update("jax_enable_x64", True)
from functools import partial

#%%
#---------------------------------------------------------------------
# Helper functions: customize sparse derivatives of K
#---------------------------------------------------------------------

# Customize jvp for functions
K_beamcol_cus = custom_jvp(BeamCol.K_beamcol_jvp,nondiff_argnums=(2,3,4)) #customized jvp 

@K_beamcol_cus.defjvp
def K_beamcol_jvp_fwd(cnct,ndof,n_bc,primals,tangents):
    '''
    Forward pass of the global stiffness contribution of beam-columns in JAX's sparse BCOO format.
    This is the customized VJP version, i.e., not naively implementing the AD feature but rather taking advantage of the sparsity.
    

    Parameters:
    ----------
    node_crds:
        ndarray storing nodal coordinates, shape of (n_node,3)

    prop:
        ndarray storing all sectional properties, shape 0 is n_beamcol

    cnct:
        ndarray storing the connectivity matrix, shape of (n_beamcol,2)
    
    ndof:
        number of dof in the system, int
    
    n_bc:
        number of beamcol in the system, int
    
    Returns:
    ------
    K:
        jax.experimental.sparse.BCOO matrix
    '''
    node_crds,prop = primals
    node_crds_dot,prop_dot= tangents
    primal_out = K_beamcol_cus(node_crds,prop,cnct,ndof,n_bc) #Forward primal call
    
    # 1D array of sectional properties
    Es = prop[:,0] # Young's modulus
    Gs = prop[:,1] # Shear modulus
    Iys = prop[:,2] # Moment of inertia
    Izs = prop[:,3] # Moment of inertia
    Js = prop[:,4] # Polar moment of inertia
    As = prop[:,5] # Sectional area

    # Connectivity matrix
    i_nodeTags = cnct[:,0] # i-node tags
    j_nodeTags = cnct[:,1] # j-node tags

    # Convert nodal coordinates to beamcol coordinates
    i_crds = node_crds[i_nodeTags,:] #shape of (n_beamcol,3)
    j_crds = node_crds[j_nodeTags,:] #shape of (n_beamcol,3)
    e_crds = jnp.hstack((i_crds,j_crds)) #shape of (n_beamcol,6)


    d_eleK_d_ele = jacfwd(BeamCol.element_K_beamcol,argnums=(0,1,2,3,4,5,6)) #function, take jacobian of elemental stiffness matrix
    vmap_d_eleK_d_ele = vmap(d_eleK_d_ele) #vmapped

    dK_e_crds,dK_Es,dK_Gs,dK_Iys,dK_Izs,dK_Js,dK_As = vmap_d_eleK_d_ele(e_crds,Es,Gs,Iys,Izs,Js,As)  #Tuple of 7, each element of tuple is shape of (n_beamcol,12,12,6) for e_crds or (n_beamcol,12,12)
    

    #Transforming data from dK_e_crds into a BCOO matrix
    dcrds_indices = vmap(BeamCol.dK_dcrds_indices)(i_nodeTags,j_nodeTags) #shape(n_beamcol,864,4), the indices of where the contribution to K is from
    dK_e_crds = (dK_e_crds.reshape(-1,864)).reshape(-1,order='F')  #data for BCOO
    dcrds_indices = dcrds_indices.reshape(-1,4,order='F') #indices for BCOO
    dK_d_node_crds = sparse.BCOO((dK_e_crds,dcrds_indices),shape=(ndof,ndof,int(ndof/6),3)) #derivatives of K wrt nodal coordinates

    #Trnsforming all the other data into a BCOO matrix for Prop
    dK_prop = jnp.stack((dK_Es,dK_Gs,dK_Iys,dK_Izs,dK_Js,dK_As),axis=3) #shape of (n_beamcol,12,12,6),stack the data
    dK_prop = (dK_prop.reshape(-1,864)).reshape(-1,order='F') #data for BCOO
    dprop_indices = vmap(BeamCol.dK_dprop_indices)(jnp.linspace(0,n_bc-1,n_bc,dtype='int32'),i_nodeTags,j_nodeTags) #shape(n_beamcol,864,4), the indices of where the contribution to K is from
    dprop_indices = dprop_indices.reshape(-1,4,order='F') #indices for BCOO
    
    dK_d_prop = sparse.BCOO((dK_prop,dprop_indices),shape=(ndof,ndof,n_bc,6)) #derivatives of K wrt element properties

    #jvp for all
    dK_d_node_crds = dK_d_node_crds.reshape(ndof,ndof,3*int(ndof/6))#reshape
    dK_d_prop = dK_d_prop.reshape(ndof,ndof,n_bc*6)#reshape
    node_crds_dot = node_crds_dot.reshape(3*int(ndof/6))#reshape
    prop_dot = prop_dot.reshape(n_bc*6)#reshape

    tangent_out = dK_d_node_crds@node_crds_dot +  dK_d_prop@prop_dot
    primal_out = primal_out.sort_indices()
    primal_out = primal_out.sum_duplicates(nse=144*n_bc)
    return primal_out, sparse.BCOO.fromdense(tangent_out,nse=144*n_bc)#TODO:AttributeError: 'UndefinedPrimal' object has no attribute 'ndim'


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
        #K = K + K_beamcol_cus(node_crds,prop_beamcols,cnct_beamcols,ndof,n_beamcol), this is still TODO.
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

