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
from jax import vmap,jit,jacfwd,jacrev,grad,value_and_grad
from jax.experimental import sparse
import jax
jax.config.update("jax_enable_x64", True)
from functools import partial
import jax.tree_util

from . import assemblemodel,solver

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
        #self.upper_bound = upper_bound
        #self.lower_bound = lower_bound

class ElementParameter():
    '''
    A class for the pararmeters of the optimization for elements.

    Parameters
    --------
    ele_tag: int
        The element tag.
    
    ele_type: int
        'BeamCol':0
        'Quad':1
    
    prop_type: int
        Which element property is the parameter.
        For Quad: int from 0 to 4 (t, E, nu, kx_mod, ky_mod)
        For Beamcol: int from 0 to 5 (E, G, Iy, Iz, J, A)

    upper_bound: float
        Upper bound of this parameter
    
    lower_bound: float
        Lower 
    '''
    def __init__(self,eletag,ele_type = 0,prop_type = 0,upper_bound=None,lower_bound=None):

        self.tag = eletag
        self.type = ele_type
        self.prop = prop_type
        #self.upper_bound = upper_bound
        #self.lower_bound = lower_bound

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
        self.eleparameters_tags = [] #list containing the nodal tags of node parameters
        self.ele_types = [] #list containing the types of element
        self.eleparameters_props = [] #list: identifiers for which property of the element is the parameter

        
        self.objective = None #objective function
        self.objective_args = None # auxiliary arguments for the objective function (user defined) 
    #----------------------------------
    # Node parameters
    #----------------------------------
    def add_nodeparameter(self,nodeparameter):
        '''
        Add a nodeparameter to SSO_Model.
        '''
        self.nodeparameters_tags.append(nodeparameter.tag)
        self.nodeparameters_xyzs.append(nodeparameter.XYZ)
        #TODO: upper bound, lower bound, etc.
    
    def add_eleparameter(self,eleparameter):
        '''
        Add a eleparameter to SSO_Model.
        '''
        self.eleparameters_tags.append(eleparameter.tag)
        self.ele_types.append(eleparameter.type)
        self.eleparameters_props.append(eleparameter.prop)
        #TODO: upper bound, lower bound, etc.

    def initialize_nodeparameters_values(self):
        '''
        Initialize the SSO_model that is ready for sensitivity analysis
        '''
        self.model.model_ready()#make model ready
        self.nodeparameters_values = jnp.array(self.model.crds[self.nodeparameters_tags,self.nodeparameters_xyzs],dtype=float)


    def initialize_eleparameters_values(self):
        '''
        Initialize the SSO_model that is ready for sensitivity analysis
        '''
        self.model.model_ready()#make model ready

        props  = (self.model.prop_beamcols,self.model.prop_quads) #Element properties
        eleparameters_list = [props[i][j,k] for (i,j,k) in zip(self.ele_types,self.eleparameters_tags,self.eleparameters_props)] #List comprehession getting the values
        self.eleparameters_values = jnp.array(eleparameters_list,dtype=float)
        self.parameters_bc = jnp.where(jnp.array(self.ele_types)==0)[0] #Indices for beamcols
        self.parameters_quads = jnp.where(jnp.array(self.ele_types)==1)[0] #Indices for quads

    def initialize_parameters_values(self):
        '''
        Initialize the SSO_model that is ready for sensitivity analysis.
        Concatenate all parameters together into a single array.
        '''
        self.initialize_nodeparameters_values()
        self.initialize_eleparameters_values()
        self.n_node_params = len(self.nodeparameters_tags) #number of node parameters
        self.n_ele_params = len(self.eleparameters_tags) #number of element parameters
        self.n_bc_params = self.parameters_bc.shape[0] #number of element parameters (beamcols)
        self.n_quad_params = self.parameters_bc.shape[0] #number of element parameters (quads)
        self.parameter_values = jnp.concatenate((self.nodeparameters_values,self.eleparameters_values))


    def update_nodeparameter(self,params_values):
        '''
        Update nodeparameter values for sso_model
        '''
        #Update in sso_model
        self.nodeparameters_values = jnp.array(params_values)
        self.parameter_values = self.parameter_values.at[:self.n_node_params].set(params_values)
    
    def update_eleparameter(self,params_values):
        '''
        Update eleparameter values for sso_model
        '''
        #Update in sso_model
        self.eleparameters_values = jnp.array(params_values)
        self.parameter_values = self.parameter_values.at[self.n_node_params:].set(params_values)
    
    def update_parameter(self,params_values):
        '''
        Update parameter values for both model and sso_model
        '''
        self.parameter_values = jnp.array(params_values)

    def update_model_parameter(self):
        #Update in model using tree_map
        #TODO, THIS IS A WORKAROUND COZ IF PARAMS_VALUES IS A TRACED ARRAY, 'TO_LIST' HERE MESSES THINGS UP
        jax.tree_util.tree_map(self.model.update_node,self.nodeparameters_tags,self.nodeparameters_xyzs,self.nodeparameters_values_untraced.tolist())
        self.model.model_ready() #let the model ready
        
    #-------------------------------------------------------------------------
    #Some methods: from parameters to results
    #-------------------------------------------------------------------------
    def node_params_crds(self,nodeparameter_values):
        '''
        Helper function that takes node parameters and outputs the new coordinates of the vertices.
        '''
        node_crds = self.model.crds.at[self.nodeparameters_tags,self.nodeparameters_xyzs].set(nodeparameter_values) #update nodal coordinates      
        return node_crds

    def ele_params_props(self,eleparameter_values):
        '''
        Helper function that takes element parameters and outputs the new properties
        '''
        ele_bc = self.parameters_bc #Indices for beamcols
        prop_beamcols = self.model.prop_beamcols.at[jnp.array(self.eleparameters_tags,dtype=int)[ele_bc],jnp.array(self.eleparameters_props,dtype=int)[ele_bc]].set(eleparameter_values[ele_bc])
        ele_quads = self.parameters_quads #Indices for quads
        prop_quads =  self.model.prop_quads.at[jnp.array(self.eleparameters_tags,dtype=int)[ele_quads],jnp.array(self.eleparameters_props,dtype=int)[ele_quads]].set(eleparameter_values[ele_quads])
        return prop_beamcols,prop_quads

    @partial(jit,static_argnums=(0,2,3)) 
    def params_u(self,parameter_values,which_solver,enforce_scipy_sparse):
        '''
        Helper function that takes parameter values and outputs displacement vector.
        '''
        #Parameters to Stiffness Matrix
        if self.n_node_params>0:
            nodeparameter_values = parameter_values[:self.n_node_params]
            node_crds = self.node_params_crds(nodeparameter_values) #Node parameters
        else:
            node_crds = self.model.crds 
        
        if self.n_ele_params>0:
            eleparameter_values = parameter_values[self.n_node_params:]
            prop_beamcols,prop_quads = self.ele_params_props(eleparameter_values) #Element parameters
        else:
            prop_beamcols,prop_quads = (self.model.prop_beamcols,self.model.prop_quads)

        #Assemble the Stiffness Matrix
        K_aug = assemblemodel.K_aug_func(node_crds,self.model.ndof,self.model.known_id,self.model.n_beamcol,self.model.cnct_beamcols,prop_beamcols,
                self.model.n_quad,self.model.cnct_quads,prop_quads) #Augmented stiffness matrix
        f_aug = assemblemodel.f_aug_func(self.model.nodal_loads,self.model.known_id) #Augmented loading vector
        
        solver_fea =self.model.select_solver(which_solver,enforce_scipy_sparse) #Select solver
        u = solver_fea(K_aug,f_aug)[:self.model.ndof]      
        return u

    @partial(jit,static_argnums=(0,2,3)) 
    def params_c(self,parameter_values,which_solver,enforce_scipy_sparse):
        '''
        Helper function that takes parameter values and outputs displacement vector.
        '''

        f_aug = assemblemodel.f_aug_func(self.model.nodal_loads,self.model.known_id) #Augmented loading vector
        u = self.params_u(parameter_values,which_solver,enforce_scipy_sparse)     
        return 0.5*f_aug[:self.model.ndof]@u

    @partial(jit,static_argnums=(0,2,3)) 
    def node_params_c(self,nodeparameter_values,which_solver,enforce_scipy_sparse):
        '''
        Helper function that takes node parameters and outputs the strain energy
        '''
        f_aug = assemblemodel.f_aug_func(self.model.nodal_loads,self.model.known_id) #Augmented loading vector
        u = self.node_params_u(nodeparameter_values,which_solver,enforce_scipy_sparse)
        
        return 0.5*f_aug[:self.model.ndof]@u
    
    #-------------------------------------------------------------------------
    #Some methods: define objective function
    #-------------------------------------------------------------------------

    def set_objective(self,objective='strain energy',func=None,func_args=None):
        '''
        Set objective function for sso_model.
        A callable f(sso_model,u,*args).

        We offer the following:
        1. total strain energy
        
        User can also define their own objective function as a callable.

        Parameters:
            objective: str
                'strain energy': the total strain energy as the objective
                'user': user defined function as the objective function
            
            func: none|callable
                if objective = 'user', func is a callable function f(sso_model,u,*args) that returns a scalar-valued objecitve.
                Please note that *args can not be dependent on u. TODO?
            
            func_args: None|Tuple
                optional arguments that will be passed to func
        '''
        if objective == 'strain energy':
            def SE(sso_model,u):
                f_aug = assemblemodel.f_aug_func(sso_model.model.nodal_loads,sso_model.model.known_id) #Augmented loading vector
                return 0.5*f_aug[:sso_model.model.ndof]@u
            self.objective = SE

        if objective == 'user':
            self.objective = func
            if func_args != None:
                self.objective_args = func_args
    

    #-------------------------------------------------------------------------
    #Some methods: from parameters to the objective
    #-------------------------------------------------------------------------

    def helper_params_to_objective(self,parameter_values,which_solver,enforce_scipy_sparse):
        '''
        From parameters to the objective.
        '''
        u = self.params_u(parameter_values,which_solver,enforce_scipy_sparse)
        if self.objective_args==None:
            return self.objective(self,u)
        else:
            return self.objective(self,u,*self.objective_args)

    def params_to_objective(self,which_solver='sparse',enforce_scipy_sparse=True):
        '''
        Calculate the objective based on parameters
        '''
        return self.helper_params_to_objective(self.parameter_values,which_solver,enforce_scipy_sparse)

    def grad_params(self,which_solver='sparse',enforce_scipy_sparse=True):
        '''
        Calculate the gradient of the objective wrt to parameters
        '''       
        return grad(self.helper_params_to_objective,argnums=(0))(self.parameter_values,which_solver,enforce_scipy_sparse)

    def value_grad_params(self,which_solver='sparse',enforce_scipy_sparse=True):
        '''
        Calculate the value and the gradient of the objective wrt to parameters
        '''        
        return value_and_grad(self.helper_params_to_objective,argnums=(0))(self.parameter_values,which_solver,enforce_scipy_sparse)




'''       def params_to_objective(parameter_values):
            u = self.node_params_u(parameter_values,which_solver,enforce_scipy_sparse)
            if self.objective_args==None:
                return self.objective(self,u)
            else:
                return self.objective(self,u,*self.objective_args)'''
'''
        def params_to_objective(parameter_values):
            u = self.node_params_u(parameter_values,which_solver,enforce_scipy_sparse)
            if self.objective_args==None:
                return self.objective(self,u)
            else:
                return self.objective(self,u,*self.objective_args)
'''