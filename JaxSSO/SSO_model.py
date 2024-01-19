'''
The optimization module of JAX-FEA: "SSO" for structural shape optimization.

Setting the parameters.
Update the finite element model using the parameters.
Set the loss function.
Calculate the sensitivity.
Conduct the optimization
'''
#%%
import numpy as np
 

import jax.numpy as jnp
from jax import vmap,jit,jacfwd,jacrev,grad
from jax.experimental import sparse
import jax
jax.config.update("jax_enable_x64", True)
from functools import partial

from . import mechanics,solver

#------------------------------------------
# Parameter types
# Node coordinate, X
# Node coordinate, Y
# Node coordinate, Z
# ... more to come
#------------------------------------------

#%%
class NodeParameter():
    '''
    A class for the pararmeters of the optimization for nodes.

    Parameters
    --------
    nodetag: int
        The node tag.
    
    X_Y_Z: int
        Which node-coordinate is the parameter.
        0:X, 1:Y, 2:Z
    
    upper_bound: float
        Upper bound of this parameter
    
    lower_bound: float
        Lower 
    '''
    def __init__(self,nodetag,X_Y_Z=2,upper_bound=None,lower_bound=None):

        self.tag = nodetag
        self.XYZ = X_Y_Z
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
    
class SSO_model():
    """
    Using optimization parameters to update the FE model.

    Parameters:
        1. Nodal coordinates
        2. Connectivity for elements
        3. Elemental properties
    SSO_model inherits these properties and update them based on the design parameters

    Conduct FEA and adjoint sensitivty analysis.
    """
    def __init__(self,model):
        '''
        Initialize a 'SSO_model.SSO_model' object, based on the initial 'model.Model' object.
        '''

        self.model  = model #Model

        #Attributes related to optimization parameters
        self.nodeparameters_tags = [] #list containing the nodal tags of node parameters
        self.nodeparameters_xyzs = [] #list corresponding to the nodal tags, takes (0,1,2), indicating which (X,Y,Z) coordinate is the parameter
        #self.nodeparameters_upbound = [] #list containing the upper bound of the parameter
        #self.nodeparameters_lowerbound = [] #list containing the lower bound of the parameter

    def add_nodeparameter(self,nodeparameter):
        '''
        Add a nodeparameter to SSO_Model.
        '''
        self.nodeparameters_tags.append(nodeparameter.tag)
        self.nodeparameters_xyzs.append(nodeparameter.XYZ)
        #TODO: upper bound, lower bound, etc.
    
    def initialize_nodeparameters_values(self):
        '''
        Initialize the SSO_model that is ready for sensitivity analysis
        '''
        self.model.model_ready()#make model ready
        self.nodeparameters_values = self.model.crds[self.nodeparameters_tags,self.nodeparameters_xyzs] 

    #-------------------------------------------------------------------------
    #Some methods: from parameters to results
    #-------------------------------------------------------------------------
    def node_params_u(self,nodeparameter_values,which_solver,enforce_scipy_sparse):
        '''
        Helper function that takes node parameters and outputs displacement vector.
        '''
        node_crds = self.model.crds.at[self.nodeparameters_tags,self.nodeparameters_xyzs].set(nodeparameter_values) #update nodal coordinates

        K_aug = mechanics.K_aug_func(node_crds,self.model.ndof,self.model.known_id,self.model.n_beamcol,self.model.cnct_beamcols,self.model.prop_beamcols,
                self.model.n_quad,self.model.cnct_quads,self.model.prop_quads) #Augmented stiffness matrix
        f_aug = mechanics.f_aug_func(self.model.nodal_loads,self.model.known_id) #Augmented loading vector
        
        #Which solver to use
        if which_solver == 'dense':
            u = solver.jax_dense_solve(K_aug,f_aug)[:self.model.ndof]
        elif which_solver == 'sparse':
            if enforce_scipy_sparse:
                u = solver.sci_sparse_solve(K_aug,f_aug)[:self.model.ndof]
            else:
                if jax.default_backend() == 'gpu':
                    u = solver.jax_sparse_solve(K_aug,f_aug)[:self.model.ndof]
                elif jax.default_backend() == 'cpu':
                    u = solver.sci_sparse_solve(K_aug,f_aug)[:self.model.ndof]        
        return u
    
    def node_params_c(self,nodeparameter_values,which_solver,enforce_scipy_sparse):
        '''
        Helper function that takes node parameters and outputs the strain energy
        '''
        f_aug = mechanics.f_aug_func(self.model.nodal_loads,self.model.known_id) #Augmented loading vector
        u = self.node_params_u(nodeparameter_values,which_solver,enforce_scipy_sparse)
        
        return 0.5*f_aug[:self.model.ndof]@u

    def grad_c_node(self,which_solver='sparse',enforce_scipy_sparse = True):
        '''
        Sensitivity of the strain energy wrt parameters
        '''
        return grad(self.node_params_c,argnums=0)(self.nodeparameters_values,which_solver,enforce_scipy_sparse)


        


    




