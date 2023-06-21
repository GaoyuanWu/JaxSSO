"""
References
---------
1. PyNite: An open-source FEA solver for python; https://github.com/JWock82/PyNite
2. Bathe, K. J. (2006). Finite element procedures. Klaus-Jurgen Bathe.
"""

#Jax and standard numpy
from jax import jit,vmap,jacfwd
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse #sparse module of Jax, under active development
from scipy.sparse import csr_matrix,linalg #solving the sparse linear system
#Partial
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)


class SSO_Model():
    """
    Using optimization parameters to update the FE model.

    Conduct FEA and adjoint sensitivty analysis.
    """

    def __init__(self,model,design_dofs):
        """
        Initialize a 'SSO_Model()' object from 'Model()'.

        model: Model() object
            The original FE model
        
        design_dofs: ndarray
            dtype int

        ini_parameters: ndarray
            same shape as 'design_dofs'

        """
        self.model = model #original FEA model
        self.design_dofs = design_dofs #Indices of coordinates to be designed. Format of (nodetag,x/y/z)
        self.parameters = None #Design parameters
        self.u = None #Displacement

    def update_param(self,params):
        '''
        Update the design parameters in the SSO model.
        '''
        self.parameters = params
    
    @partial(jit,static_argnums=(0))
    def x_to_K(self,parameter_values):
        cnct_bc = self.model.cnct_beamcols() #connectivity
        i_nodes = cnct_bc[:,0] # i-node tags
        j_nodes = cnct_bc[:,1] # j-node tags
        
        og_crds = self.model.nodes_crds() #coordinates
        crds = jnp.asarray(og_crds)
        crds = crds.at[self.design_dofs[:,0],self.design_dofs[:,1]].set(parameter_values)  # update the parameter
        i_crds = crds[i_nodes,:] #shape of (n_beamcol,3)
        j_crds = crds[j_nodes,:] #shape of (n_beamcol,3)
        e_crds = jnp.hstack((i_crds,j_crds)) #shape of (n_beamcol,6)

        cs_prop = self.model.beamcols_cross_prop() #sectional properties
        K = self.model.K_beamcol(e_crds,cs_prop) #global stiffness matrix in BCOO
        return K
    

    @partial(jit,static_argnums=(0))
    def x_to_K_data(self,parameter_values):
        return self.x_to_K(parameter_values).data
    
    @partial(jit,static_argnums=(0))
    def dK_dx(self,parameter_values):
        '''
        The sensitivity of global stiffness matrix wrt design parameters
        '''
        return jacfwd(self.x_to_K_data)(parameter_values)

    def solve(self):
        '''
        Solve FEA for u.
        '''
        K_G = self.x_to_K(self.parameters) # The updated stiffness matrix
        u_G = self.model.scipy_sparse_solve(K_G) # The updated displacement
        self.u = u_G

    def Compliance(self):
        '''
        The compliance: 0.5@f.T@u.
        The objective function.
        '''
        return 0.5*np.dot(self.model.f,self.u)

    def Sensitivity(self):
        '''
        The sensitivity using the adjoint method.
        '''
        dKdx = self.dK_dx(self.parameters) #array shape of (n_beamcol*144,n_design_param)
        rows,cols = self.model.vmapped_rows_cols() #Get rows and columns
        rows = rows.ravel()
        cols = cols.ravel()
        u_r = self.u[rows] #displacement corresponding to rows
        u_c = self.u[cols] #displacement corresponding to cols
        return -0.5*(jnp.multiply(u_r,u_c)@dKdx)





    