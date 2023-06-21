"""
References
---------
1. PyNite: An open-source FEA solver for python; https://github.com/JWock82/PyNite
2. Bathe, K. J. (2006). Finite element procedures. Klaus-Jurgen Bathe.
"""

#Jax and standard numpy
from jax import jit,vmap
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.experimental import sparse
from scipy import sparse as sci_sparse
import scipy.sparse.linalg as sci_linalg 
import jax
jax.config.update("jax_enable_x64", True)
#JaxSSO
from .Node import Node #Node() class
from .BeamCol import BeamCol #BeamCol() class

class Model():
    """
    The FEA model (pre-processing) yet to be analyzed.
    Including:
        nodes
        elements:
            beam-columns
        boundaryconditions
        nodal_load
    """

    def __init__(self):
        """
        Initialize a 'Model' object.

        """
        
        self.nodes = {}    # Nodes in the system
        self.beamcols = {}    # Beam-columns in the system
        self.fixed_dofs = [] #Storing the indices of the fixed dofs
        self.f = None #nodal loads, shape of (n_dof,)


    def node(self, nodeTag, X, Y, Z):
        """
        Adding/updating a node to the model.

        Inputs:
        -----
        nodeTag: int 
            the tag/index of this node

        X, Y, Z: float
            the coordinates of thie node

        """
        
        # Create a new node
        new_node = Node(nodeTag, X, Y, Z)
        
        # Add the new node/updating an existing node to model
        self.nodes[nodeTag] = new_node

    def beamcol(self, eleTag, i_nodeTag, j_nodeTag, E, G, Iy, Iz, J, A):
        '''
        Adding/updating a beam-column element to the model
        
        Inputs
        ----------
        eleTag : int
            Index of this element

        i_nodeTag : int
            The tag of the i-node (start node).

        j_nodeTags : int
            The tag of the j-node (end node).

        E: float
            Young's modulus

        G: float
            Shear modulus

        Iy,Iz: float
            Moment of inertia about local y and local z axis

        J: float
            Torsional moment of inertia of cross section

        A: float
            Area of cross section
        '''
        node_i = self.nodes[i_nodeTag] #first node
        node_j = self.nodes[j_nodeTag] #second node
        ele_crds = [node_i.X,node_i.Y,node_i.Z,node_j.X,node_j.Y,node_j.Z] #coordinates

        #Create/update a beam-column and save it to the dictionary
        new_beamcol = BeamCol(eleTag, node_i.nodeTag, node_j.nodeTag, ele_crds, E, G, Iy, Iz, J, A)
        self.beamcols[eleTag] = new_beamcol


    def add_support(self,node_index,active_supports=[1,1,1,1,1,1]):
        '''
        Adding nodal support in the model.
        Inputs
        -----
        node_index: int
            The index of the node.
        
        active_supports: list, 1darray
            Whether the support for [X,Y,Z,RX,RY,RZ] is active.
            1 is active, 0 is deactivated.
            Default is fixed.
        '''
        dofs = np.linspace(node_index*6,node_index*6+5,6,dtype=int)
        active_supports = np.array(active_supports,dtype=int) #convert to standard np.array
        where_active = np.argwhere(active_supports==1).ravel() #which dofs are active
        self.fixed_dofs.append(dofs[where_active]) #append to the global list

    def add_nodal_load(self,f):
        '''
        Add the global load vector {f}.
        Inputs
        ---
        f: ndarray of shape (6*n_node)
            The global nodal loading vector.
        '''
        self.f = f 
    
    def bc_indices(self):
        '''
        Return the indices of unknown and known dof-displacement based on the boundary conditions.
        Returns
        ------
        known_indices: ndarray 
            indices of displacement that are known. 
        
        unknown_indices: ndarray
            indices of unknown displacement
        '''
        known_indices = (np.array(self.fixed_dofs,dtype=int)).ravel() #convert the active support indices list to np.array
        all_indices = np.linspace(0,6*len(self.nodes)-1,6*len(self.nodes),dtype=int) #create a container for all indices
        unknown_indices = all_indices[np.where(np.isin(all_indices,known_indices,assume_unique=True,invert=True))] #slice out the known indices 
        return known_indices,unknown_indices
    
    def nodes_tags(self):
        '''
        Get all the nodal tags in the structure.
        List of length (n_node).
        '''
        return [ node.nodeTag for node in self.nodes.values()]
    
    def nodes_xs(self):
        '''
        Get all the nodal x-coordinates in the structure.
        List of length (n_node).
        '''
        return [ node.X for node in self.nodes.values()]
    
    def nodes_ys(self):
        '''
        Get all the nodal y-coordinates in the structure.
        List of length (n_node).
        '''
        return [ node.Y for node in self.nodes.values()]
    
    def nodes_zs(self):
        '''
        Get all the nodal z-coordinates in the structure.
        List of length (n_node).
        '''
        return [node.Z for node in self.nodes.values()]
    
    def nodes_crds(self):
        '''
        Return a 2D array of shape (n_node,3) for nodal coordinates.
        '''
        col_1 = np.expand_dims(np.array(self.nodes_xs()),axis=1)
        col_2 = np.expand_dims(np.array(self.nodes_ys()),axis=1)
        col_3 = np.expand_dims(np.array(self.nodes_zs()),axis=1)
        return np.hstack((col_1,col_2,col_3))
    
    def beamcols_tags(self):
        '''
        Get all the ele tags in the structure.
        List of length (n_node).
        '''
        return np.array([bc.eleTag for bc in self.beamcols.values()])
    
    def cnct_beamcols(self):
        '''
        Get the connectivity matrix of the structure.
        Return a 2D array of shape (n_beamcol,2)
        '''
        i_nodes_tags = np.array([[bc.i_nodeTag for bc in self.beamcols.values()]],dtype=int)
        j_nodes_tags = np.array([[bc.j_nodeTag for bc in self.beamcols.values()]],dtype=int)
        return np.vstack((i_nodes_tags,j_nodes_tags)).T
    
    def beamcols_cross_prop(self):
        '''
        Get cross-sectional properties: E, G, Iy, Iz, J, A
        Return a 2D array of shape (n_beamcol,5)
        '''
        Es = np.array([[bc.E for bc in self.beamcols.values()]])
        Gs = np.array([[bc.G for bc in self.beamcols.values()]])
        Iys = np.array([[bc.Iy for bc in self.beamcols.values()]])
        Izs = np.array([[bc.Iz for bc in self.beamcols.values()]])
        Js = np.array([[bc.J for bc in self.beamcols.values()]])
        As = np.array([[bc.A for bc in self.beamcols.values()]])
        return np.vstack((Es, Gs, Iys, Izs, Js, As)).T
    
    def nodal_loads(self):
        '''
        Return nodal loads
        '''
        return self.f
    
    
    @staticmethod
    def K_2_Global(i_nodeTag,j_nodeTag,crds,E,G,Iy,Iz,J,A):
        '''
        Assign element's stiffness matrix into the expanded global stiffness (sparse) matrix.
        Will be vmapped and applied to all elements simultaneously

            
        Return:
            data: ndarray of shape(144,)
                The values of local stiffness matrix
            
            indices: ndarray of shape (144,2)
                The corresponding indices in global stiffness matrix.
                The first column is the row number, the second coclumn is the column number

        '''
        T_K = BeamCol.T(crds) #Coordinate transformation
        K_loc = BeamCol.K_local(crds,E,G,Iy,Iz,J,A) #Local stiffness matrix
        K_eleG = BeamCol.K(T_K,K_loc) #Element's stiffness matrix in global coordinates, yet to be assigned
        data = jnp.ravel(K_eleG) #local stiffness matrix, flatten
        indices_dof = jnp.hstack((jnp.linspace(i_nodeTag*6,i_nodeTag*6+5,6,dtype=int),jnp.linspace(j_nodeTag*6,j_nodeTag*6+5,6,dtype=int))) #indices represented the dofs of this beamcol
        rows,cols = jnp.meshgrid(indices_dof,indices_dof,indexing='ij')#rows and columns
        indices = jnp.vstack((rows.ravel(),cols.ravel())).T #indices in global stiffness matrix
        return data,indices
    
    @staticmethod
    def rows_cols(i_nodeTag,j_nodeTag):
        '''
        Return the rows and cols for a beamcol
        '''
        indices_dof = jnp.hstack((jnp.linspace(i_nodeTag*6,i_nodeTag*6+5,6,dtype=int),jnp.linspace(j_nodeTag*6,j_nodeTag*6+5,6,dtype=int)))
        rows,cols = jnp.meshgrid(indices_dof,indices_dof,indexing='ij')#rows and columns
        return rows.ravel(),cols.ravel()
    
    @partial(jit, static_argnums=(0))
    def vmapped_rows_cols(self):
        cnct_bc = self.cnct_beamcols() #connectivity
        i_nodeTags = cnct_bc[:,0] # i-node tags
        j_nodeTags = cnct_bc[:,1] # j-node tags
        rows,cols = vmap(self.rows_cols,(0,0))(i_nodeTags,j_nodeTags)
        return rows,cols

    @partial(jit, static_argnums=(0))
    def K_beamcol(self,e_crds,sectional_prop):
        '''
        Return the global stiffness matrix stored in sparse matrix format.
        This stiffness matrix is without the consideration of boundary conditions.

        Vmap BeamCol.K_2_Global(i_nodeTag,j_nodeTag,crds,E,G,Iy,Iz,J,A)

        Parameters:
            n_dof: int
                number of dofs in the system, 6*n_node
            i_nodeTags: tuple of length (n_beamcol)
                connectivity relationship of the element
            j_nodeTags: tuple of length (n_beamcol)
                connectivity relationship of the element
            e_crds: ndarray of shape (n_beamcol,6)
                all the nodal coordinates
            sectional_prop: ndarray of shape (n_beamcol,5)
                sectional properties of beam-columns, each row for (E, G, Iy, Iz, J, A) 
        '''
        ##Something non-static
        Es = sectional_prop[:,0] # Young's modulus
        Gs = sectional_prop[:,1] # Shear modulus
        Iys = sectional_prop[:,2] # Moment of inertia
        Izs = sectional_prop[:,3] # Moment of inertia
        Js = sectional_prop[:,4] # Young's modulus
        As = sectional_prop[:,5] # Young's modulus
        

        ##Something static
        n_dof = len(self.nodes)*6 #number of dofs
        cnct_bc = self.cnct_beamcols() #connectivity
        i_nodeTags = cnct_bc[:,0] # i-node tags
        j_nodeTags = cnct_bc[:,1] # j-node tags

        data,indices = vmap(self.K_2_Global)(i_nodeTags,j_nodeTags,e_crds,Es,Gs,Iys,Izs,Js,As)  #Get the indices and values storing all the local stiffness matrices
        data = data.reshape(-1) # re-dimension the data to 1darray (144*n_beamcol,)
        indices = indices.reshape(-1,2) #re-dimension to 2darray of shape (144*n_beamcol,2)
        K_global = sparse.BCOO((data,indices),shape=(n_dof,n_dof))

        return K_global
    


    @partial(jit, static_argnums=(0))
    def jax_sparse_solve(self,K_11):
        '''
        Solve the linear system.
        
        Parameters:
            
            K: jax.experimental.sparse.BCOO
                Global stiffness matrx


        '''
        known_dofs,unknown_dofs = self.bc_indices() #indices of known and unknown displacement

        #Convert to CSR formart from BCOO        
        ndof = unknown_dofs.shape[0] + known_dofs.shape[0]

        #K_11 = K[unknown_dofs,:][:,unknown_dofs] # JAX.SPARSE does not support gathering
        K_11_csr = sparse.CSR((K_11.data, K_11.indices[:,0], K_11.indices[:,1]), shape=(ndof,ndof)) #convert to CSR format

        #Load vector @ unknown indices
        f_uk = self.f[unknown_dofs]

        #Solving FEA
        u_unknown = sparse.linalg.spsolve(K_11_csr.data,K_11_csr.indices, K_11_csr.indptr,f_uk) #Experimental Jax function for spsolve, under Google development: https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html
        
        #Global displacement vector
        u_all = jnp.zeros(6*len(self.nodes))
        u_all = u_all.at[unknown_dofs].set(u_unknown)

        return u_all
    
    def scipy_sparse_solve(self,K):
        '''
        Solve the linear system.
        
        Parameters:
            
            K: jax.experimental.sparse.BCOO
                Global stiffness matrx
        '''
        known_dofs,unknown_dofs = self.bc_indices() #indices of known and unknown displacement

        #Convert to CSR formart from BCOO
        ndof = unknown_dofs.shape[0] + known_dofs.shape[0]
        K_csr = sci_sparse.csr_matrix((K.data, (K.indices[:,0], K.indices[:,1])), shape=(ndof,ndof)) #convert to CSR format
        K_11_csr = K_csr[unknown_dofs,:][:,unknown_dofs] # K11, unknown indices

        #Load vector @ unknown indices
        f_uk = self.f[unknown_dofs]

        #Solving FEA
        u_unknown = sci_linalg.spsolve(K_11_csr,f_uk) #Scipy.sparse.spsolve
        
        #Global displacement vector
        u_all = jnp.zeros(6*len(self.nodes))
        u_all = u_all.at[unknown_dofs].set(u_unknown)

        return u_all
    
    @partial(jit, static_argnums=(0))
    def K(self):
        cnct_bc = self.cnct_beamcols() #connectivity
        i_nodes = cnct_bc[:,0] # i-node tags
        j_nodes = cnct_bc[:,1] # j-node tags
        
        crds = self.nodes_crds() #coordinates
        i_crds = crds[i_nodes,:] #shape of (n_beamcol,3)
        j_crds = crds[j_nodes,:] #shape of (n_beamcol,3)
        e_crds = jnp.hstack((i_crds,j_crds)) #shape of (n_beamcol,6)

        cs_prop = self.beamcols_cross_prop() #sectional properties
        K = self.K_beamcol(e_crds,cs_prop) #global stiffness matrix in BCOO
        return K

    #def solve_jax(self):
    #    return self.jax_sparse_solve(self.K_11())
    
    def solve_scipy(self):
        return self.scipy_sparse_solve(self.K())
