'''
This modules take 'model' object as input and assemble the linear system Ax=b(Ku=f) for the solver to solve.
'''
#%%
import numpy as np
import jax.numpy as jnp
from .element import BeamCol,Quad,Truss
from jax import vmap,jit
from jax.experimental import sparse
import jax
jax.config.update("jax_enable_x64", True)
from functools import partial
#%%
#---------------------------------------------------------------------
# Helper functions: beam-columns
#---------------------------------------------------------------------

def element_K_beamcol(crds,E,G,Iy,Iz,J,A):
    '''
    Given the attributes of a beam-column. Return element's stiffness matrix in global coordinate system.
    The element's stiffness matrix is flattened (raveled).
    '''
    ele_K_local = BeamCol.K_local(crds,E,G,Iy,Iz,J,A) # Element's stiffness matrix, before coordinate trnasformation
    ele_T = BeamCol.T(crds) #Coordinate transormation matrix
    ele_K_global = jnp.matmul(jnp.linalg.solve(ele_T,ele_K_local),ele_T)
    return ele_K_global


def element_K_beamcol_indices(i_nodeTag,j_nodeTag):
    '''
    Given the node tags of a beam-column, return the corresponding indices (rows, columns) of this beam column in the global stiffness matrix.
    '''
    indices_dof = jnp.hstack((jnp.linspace(i_nodeTag*6,i_nodeTag*6+5,6,dtype='int32'),jnp.linspace(j_nodeTag*6,j_nodeTag*6+5,6,dtype='int32'))) #indices represented the dofs of this beamcol
    rows,cols = jnp.meshgrid(indices_dof,indices_dof,indexing='ij')#rows and columns
    indices = jnp.vstack((rows.ravel(),cols.ravel())).T #indices in global stiffness matrix
    return indices

@jit
def K_local_beamcol(node_crds,prop,cnct):
    '''
    Return ndarray that stores the local stiffness matrix of all beam-columns.

    Parameters:
    ----------
    node_crds:
        ndarray storing nodal coordinates, shape of (n_node,3)

    prop:
        ndarray storing all sectional properties, shape 0 is n_beamcol

    cnct:
        ndarray storing the connectivity matrix, shape of (n_beamcol,2)

    
    Returns:
    ------
    k_local: ndarray of shape (n_beamcol,12,12)
    '''
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

    return vmap(BeamCol.K_local,in_axes=(0,0,0,0,0,0,0,))(e_crds,Es,Gs,Iys,Izs,Js,As) # Element's stiffness matrix, before coordinate trnasformation

@jit
def T_beamcol(node_crds,cnct):
    '''
    Return ndarray that stores all the coordinate transformation matrix of all beam-columns.

    Parameters:
    ----------
    node_crds:
        ndarray storing nodal coordinates, shape of (n_node,3)

    cnct:
        ndarray storing the connectivity matrix, shape of (n_beamcol,2)

    
    Returns:
    ------
    T: ndarray of shape (n_beamcol,12,12)
    '''
    # Connectivity matrix
    i_nodeTags = cnct[:,0] # i-node tags
    j_nodeTags = cnct[:,1] # j-node tags

    # Convert nodal coordinates to beamcol coordinates
    i_crds = node_crds[i_nodeTags,:] #shape of (n_beamcol,3)
    j_crds = node_crds[j_nodeTags,:] #shape of (n_beamcol,3)
    e_crds = jnp.hstack((i_crds,j_crds)) #shape of (n_beamcol,6)

    return vmap(BeamCol.T,in_axes=(0))(e_crds)


@partial(jit,static_argnums=(3))
def K_beamcol(node_crds,prop,cnct,ndof):
    '''
    Return the global stiffness contribution of beam-columns in JAX's sparse BCOO format.

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
    
    Returns:
    ------
    K:
        jax.experimental.sparse.BCOO matrix
    '''
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

    data = jnp.ravel(vmap(element_K_beamcol)(e_crds,Es,Gs,Iys,Izs,Js,As))  #Get the stiffness values, raveled
    data = data.reshape(-1) # re-dimension the data to 1darray (144*n_beamcol,)
    indices = vmap(element_K_beamcol_indices)(i_nodeTags,j_nodeTags) # Get the indices
    indices = indices.reshape(-1,2) #re-dimension to 2darray of shape (144*n_beamcol,2)
    K= sparse.BCOO((data,indices),shape=(ndof,ndof))
    return K

#---------------------------------------------------------------------
# Helper functions: quad shells
#---------------------------------------------------------------------

def element_K_quad(crds, t, E, nu, kx_mod, ky_mod):
    '''
    Given the attributes of a quad. Return element's stiffness matrix in global coordinate system.
    The element's stiffness matrix is flattened (raveled).
    '''
    ele_Km = Quad.k_m(crds, t, E, nu, Quad.loc_crds, Quad.Cm, Quad.B_m, Quad.J, Quad.index_k_m, kx_mod, ky_mod) # Element's stiffness matrix (membrane), before coordinate trnasformation.
    ele_Kb = Quad.k_b(crds, t, E, nu, Quad.Cb, Quad.Cs, Quad.J, Quad.B_kappa, Quad.loc_crds, Quad.B_gamma_MITC4, Quad.index_k_b, kx_mod=1.0, ky_mod=1.0) # Element's stiffness matrix (bending), before coordinate trnasformation.
    ele_K = ele_Km + ele_Kb #Sum bending & membrane contribution
    ele_T = Quad.T(crds) #Coordinate transormation matrix
    ele_K_global = jnp.matmul(jnp.linalg.solve(ele_T,ele_K),ele_T)
    return ele_K_global

def element_K_quad_local(crds, t, E, nu, kx_mod, ky_mod):
    '''
    Given the attributes of a quad. Return element's stiffness matrix in global coordinate system.
    The element's stiffness matrix is flattened (raveled).
    '''
    ele_Km = Quad.k_m(crds, t, E, nu, Quad.loc_crds, Quad.Cm, Quad.B_m, Quad.J, Quad.index_k_m, kx_mod, ky_mod) # Element's stiffness matrix (membrane), before coordinate trnasformation.
    ele_Kb = Quad.k_b(crds, t, E, nu, Quad.Cb, Quad.Cs, Quad.J, Quad.B_kappa, Quad.loc_crds, Quad.B_gamma_MITC4, Quad.index_k_b, kx_mod=1.0, ky_mod=1.0) # Element's stiffness matrix (bending), before coordinate trnasformation.
    ele_K = ele_Km + ele_Kb #Sum bending & membrane contribution
    return ele_K

def element_K_quad_indices(i_nodeTag,j_nodeTag,m_nodeTag,n_nodeTag):
    '''
    Given the node tags of a quad shell, return the corresponding indices (rows, columns) of this quad in the global stiffness matrix.
    '''
    indices_dof = jnp.hstack((jnp.linspace(i_nodeTag*6,i_nodeTag*6+5,6,dtype='int32'),jnp.linspace(j_nodeTag*6,j_nodeTag*6+5,6,dtype='int32'),
                    jnp.linspace(m_nodeTag*6,m_nodeTag*6+5,6,dtype='int32'),jnp.linspace(n_nodeTag*6,n_nodeTag*6+5,6,dtype='int32'))) #indices represented the dofs of this quad
    rows,cols = jnp.meshgrid(indices_dof,indices_dof,indexing='ij')#rows and columns
    indices = jnp.vstack((rows.ravel(),cols.ravel())).T #indices in global stiffness matrix
    return indices

@jit
def K_local_quad(node_crds,prop,cnct):
    '''
    Return ndarray that stores the local stiffness matrix of all beam-columns.

    Parameters:
    ----------
    node_crds:
        ndarray storing nodal coordinates, shape of (n_node,3)

    prop:
        ndarray storing all sectional properties, shape 0 is n_quad:t, E, nu, kx_mod, ky_mod

    cnct:
        ndarray storing the connectivity matrix, shape of (n_quad,4)

    
    Returns:
    ------
    k_local: ndarray of shape (n_quad,24,24)
    '''
    # 1D array of sectional properties
    ts = prop[:,0] 
    Es = prop[:,1] 
    nus = prop[:,2] 
    kx_mods = prop[:,3] 
    ky_mods = prop[:,4] 

    # Connectivity matrix
    i_nodeTags = cnct[:,0] # i-node tags
    j_nodeTags = cnct[:,1] # j-node tags
    m_nodeTags = cnct[:,2] # m-node tags
    n_nodeTags = cnct[:,3] # n-node tags

    # Convert nodal coordinates to beamcol coordinates
    i_crds = node_crds[i_nodeTags,:] #shape of (n_quad,3)
    j_crds = node_crds[j_nodeTags,:] #shape of (n_quad,3)
    m_crds = node_crds[m_nodeTags,:] #shape of (n_quad,3)
    n_crds = node_crds[n_nodeTags,:] #shape of (n_quad,3)
    e_crds = jnp.hstack((i_crds,j_crds,m_crds,n_crds)) #shape of (n_quad,12)

    return vmap(element_K_quad_local)(e_crds,ts,Es,nus,kx_mods,ky_mods)

@jit
def T_quad(node_crds,cnct):
    '''
    Return ndarray that stores all the coordinate transformation matrix of all beam-columns.

    Parameters:
    ----------
    node_crds:
        ndarray storing nodal coordinates, shape of (n_node,3)

    cnct:
        ndarray storing the connectivity matrix, shape of (n_quad,4)

    
    Returns:
    ------
    T: ndarray of shape (n_quad,24,24)
    '''
    # Connectivity matrix
    i_nodeTags = cnct[:,0] # i-node tags
    j_nodeTags = cnct[:,1] # j-node tags
    m_nodeTags = cnct[:,2] # m-node tags
    n_nodeTags = cnct[:,3] # n-node tags

    # Convert nodal coordinates to beamcol coordinates
    i_crds = node_crds[i_nodeTags,:] #shape of (n_quad,3)
    j_crds = node_crds[j_nodeTags,:] #shape of (n_quad,3)
    m_crds = node_crds[m_nodeTags,:] #shape of (n_quad,3)
    n_crds = node_crds[n_nodeTags,:] #shape of (n_quad,3)
    e_crds = jnp.hstack((i_crds,j_crds,m_crds,n_crds)) #shape of (n_quad,12)

    return vmap(Quad.T,in_axes=(0))(e_crds)

@partial(jit,static_argnums=(3))
def K_quad(node_crds,prop,cnct,ndof):
    '''
    Return the global stiffness contribution of quads in JAX's sparse BCOO format.

    Parameters:
    ----------
    node_crds:
        ndarray storing nodal coordinates, shape of (n_node,3)

    prop:
        ndarray storing all sectional properties, shape 0 is n_quad

    cnct:
        ndarray storing the connectivity matrix, shape of (n_quad,4)
    
    ndof:
        number of dof in the system, int
    
    Returns:
    ------
    K:
        jax.experimental.sparse.BCOO matrix
    '''
    # 1D array of sectional properties: t, E, nu, kx_mod, ky_mod
    ts = prop[:,0] # Thickness
    Es = prop[:,1] # Young's modulus
    nus = prop[:,2] # Poison's ratio
    kx_mods = prop[:,3] # Stiffness modification coefficient, along x
    ky_mods = prop[:,4] # Stiffness modification coefficient, along y


    # Connectivity matrix
    i_nodeTags = cnct[:,0] # i-node tags
    j_nodeTags = cnct[:,1] # j-node tags
    m_nodeTags = cnct[:,2] # m-node tags
    n_nodeTags = cnct[:,3] # n-node tags

    # Convert nodal coordinates to beamcol coordinates
    i_crds = node_crds[i_nodeTags,:] #shape of (n_quad,3)
    j_crds = node_crds[j_nodeTags,:] #shape of (n_quad,3)
    m_crds = node_crds[m_nodeTags,:] #shape of (n_quad,3)
    n_crds = node_crds[n_nodeTags,:] #shape of (n_quad,3)
    
    e_crds = jnp.hstack((i_crds,j_crds,m_crds,n_crds)) #shape of (n_quad,12)

    data = jnp.ravel(vmap(element_K_quad)(e_crds,ts,Es,nus,kx_mods,ky_mods))  #Get the stiffness values, raveled
    data = data.reshape(-1) # re-dimension the data to 1darray (576*n_quad,)
    indices = vmap(element_K_quad_indices)(i_nodeTags,j_nodeTags,m_nodeTags,n_nodeTags) # Get the indices
    indices = indices.reshape(-1,2) #re-dimension to 2darray of shape (576*n_quad,2)
    K= sparse.BCOO((data,indices),shape=(ndof,ndof))
    return K


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
    Take all the parameters and return the stiffness matrix
    '''

    #Initial stiffness in bcoo format
    data = jnp.array([0.])
    indices = jnp.array([[0,0]],dtype='int32')
    K = sparse.BCOO((data,indices),shape=(ndof,ndof))

    #Go over different element types
    if n_beamcol>0:
        K = K + K_beamcol(node_crds,prop_beamcols,cnct_beamcols,ndof)
    if n_quad>0:
        K = K + K_quad(node_crds,prop_quads,cnct_quads,ndof)
    
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
    Given a model, return the global stiffness matrix in BCOO format.
    
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

#---------------------------------------------------------------------
# Helper functions for 'sso_model' objects
#---------------------------------------------------------------------



