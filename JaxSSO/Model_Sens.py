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
from jax.experimental import sparse #sparse module of Jax, under active development

#Partial
from functools import partial

#JaxSSO
from .Node import Node #'Node' objects
from .BeamCol import BeamCol #'BeamCol' objects
from . import BeamCol as BeamColSens # Import BeamCol.py module 


class Model_Sens():
    """
    Objects for sensitivity analysis model in JaxSSO
    Including:
    1. Defining the model that is necessary for global stiffness matrix (adding nodes, elements)
    2. Assemble the global stiffness matrix(K) without considering boundary conditions
    3. Get the sensitivity of structure's global stiffness matrix 'K' w.r.t. nodal coordinates
    Not including:
    1. Assigning loads
    2. Assigning boundary conditions
    3. Solving [K]{u} = {f}
    """

    def __init__(self):
        """
        Initialize a 'Model_Sens' object.
        Create lists storing:
            1. Nodes
            2. BeamCol elements
        
        """
        
        self.nodes = {}      # A dictionary of the structure's nodes
        self.beamcols = {}    # A dictionary of the structure's beam-columns
        self.supports_indices = [] # A list storing all the indices of the active supports 



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
        
        # Add the new node/updating an existing node to model\
        self.nodes[nodeTag] = new_node

    def beamcol(self, eleTag, i_node, j_node, E, G, Iy, Iz, J, A):
        '''
        Adding/updating a beam-column element to the model
        
        Inputs
        ----------
        eleTag : int
            Index of this element

        i_node : int
            The tag of the i-node (start node).

        j_node : int
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
        # Nodes' coordinates
        x1,y1,z1 = [self.nodes[i_node].X, self.nodes[i_node].Y, self.nodes[i_node].Z]
        x2,y2,z2 = [self.nodes[j_node].X, self.nodes[j_node].Y, self.nodes[j_node].Z]

        #Create/update a beam-column and save it to the dictionary
        new_beamcol = BeamCol(eleTag, i_node, j_node, x1, y1, z1, x2, y2, z2, E, G, Iy, Iz, J, A)
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
        indices = np.linspace(node_index*6,node_index*6+5,6,dtype=int)
        active_supports = np.array(active_supports,dtype=int) #convert to standard np.array
        where_active = np.argwhere(active_supports==1).ravel() #which dofs are active
        self.supports_indices.append(indices[where_active]) #append to the global list

    def add_nodal_load(self,f):
        '''
        Add the global load vector {f}.
        Inputs
        ---
        f: ndarray of shape (6*n_node)
            The global nodal loading vector.
        '''
        self.f = f 
        
    def K(self):
        '''
        Return the global stiffness of the system in 'jax.experimental.sparce.BCOO' format.
        Without the boundary conditions assigned.
        
        '''
        n_dof = 6*len(self.nodes) #number of dof
        if self.beamcols:
            #Creating containers storing attributes
            beamcol_list = list(self.beamcols.values())  #create a list for the beamcols

            beamcol_eleTag = np.array([beamcol.eleTag for beamcol in beamcol_list])  #Tags of beamcols
            beamcol_i_nodeTag = np.array([beamcol.i_nodeTag for beamcol in beamcol_list])  #Tags of i-node of beamcols
            beamcol_j_nodeTag = np.array([beamcol.j_nodeTag for beamcol in beamcol_list])  #Tags of j-node of beamcols
            beamcol_x1 = np.array([beamcol.x1 for beamcol in beamcol_list],dtype=float) #x of inode of beamcols
            beamcol_y1 = np.array([beamcol.y1 for beamcol in beamcol_list],dtype=float) #y of inode of beamcols
            beamcol_z1 = np.array([beamcol.z1 for beamcol in beamcol_list],dtype=float) #z of inode of beamcols
            beamcol_x2 = np.array([beamcol.x2 for beamcol in beamcol_list],dtype=float) #x of jnode of beamcols
            beamcol_y2 = np.array([beamcol.y2 for beamcol in beamcol_list],dtype=float) #y of jnode of beamcols
            beamcol_z2 = np.array([beamcol.z2 for beamcol in beamcol_list],dtype=float) #z of jnode of beamcols
            beamcol_E = np.array([beamcol.E for beamcol in beamcol_list],dtype=float) #E of beamcols
            beamcol_G = np.array([beamcol.G for beamcol in beamcol_list],dtype=float) #G of beamcols
            beamcol_Iy = np.array([beamcol.Iy for beamcol in beamcol_list],dtype=float) #Iy of beamcols
            beamcol_Iz = np.array([beamcol.Iz for beamcol in beamcol_list],dtype=float) #Iy of beamcols
            beamcol_J = np.array([beamcol.J for beamcol in beamcol_list],dtype=float) #J of beamcols
            beamcol_A = np.array([beamcol.A for beamcol in beamcol_list],dtype=float) #A of beamcols

            #Create 'BeamCol' object from arrays of attributes
            beamcols_replica = BeamCol(beamcol_eleTag, beamcol_i_nodeTag, beamcol_j_nodeTag, 
                                                beamcol_x1,beamcol_y1,beamcol_z1,beamcol_x2,beamcol_y2,
                                                beamcol_z2,beamcol_E,beamcol_G,beamcol_Iy,beamcol_Iz,beamcol_J,beamcol_A)

            # Call the functions defined for 'BeamCol' objects
            # Implement jax.vmap
            data,indices = vmap(BeamColSens.Ele_K_to_Global)(beamcols_replica)  #Get the indices and values storing all the local stiffness matrices
            data = data.reshape(-1) # re-dimension the data to 1darray (144*n_beamcol,)
            indices = indices.reshape(-1,2) #re-dimension to 2darray of shape (144*n_beamcol,2)
            K_global = sparse.BCOO((data,indices),shape=(n_dof,n_dof))
        
        return K_global

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
        known_indices = (np.array(self.supports_indices,dtype=int)).ravel() #convert the active support indices list to np.array
        all_indices = np.linspace(0,6*len(self.nodes)-1,6*len(self.nodes),dtype=int) #create a container for all indices
        unknown_indices = all_indices[np.where(np.isin(all_indices,known_indices,assume_unique=True,invert=True))] #slice out the known indices 
        return known_indices,unknown_indices
        
    def solve(self):
        '''
        Solving [K]{u} = {f}
        '''
        K_dense = self.K().todense() #Convert to dense matrix
        known_indices,unknown_indices = self.bc_indices() #Get the bc indices

        #Parition the global K into four blocks
        #based on the boundary conditions
        K11 = K_dense[unknown_indices,:][:,unknown_indices] #K11
        #K12 = K_dense[unknown_indices,:][:,known_indices]
        #K21 = K_dense[known_indices,:][:,unknown_indices]
        #K22 = K_dense[known_indices,:][:,known_indices]

        #Load vector @ unknown indices
        f_uk = self.f[unknown_indices]

        #Solving FEA
        u_unknown = jnp.linalg.solve(K11,f_uk)

        #Global displacement vector
        u_all = jnp.zeros(6*len(self.nodes))
        u_all = u_all.at[unknown_indices].set(u_unknown)

        self.u = u_all #store the displacement vector



    def Sens_C_Coord(self,u):
        '''
        Return the sensitivity of the strain energy (compliance) of the system wrt the displacement vector 'u'.

        Inputs
        -----
        u: ndarray of shape (6*n_node)
            Displacement of each DOF

        Returns
        -----
        dcdx: ndarray of shape(3*n_node)
            Gradient of the strain energy wrt nodal coordinates
        '''
        dcdx_all = jnp.zeros(3*len(self.nodes)) #container storing the sensitivity
        if self.beamcols:
            
            #Creating containers storing attributes
            beamcol_list = list(self.beamcols.values())  #create a list for the beamcols

            beamcol_eleTag = np.array([beamcol.eleTag for beamcol in beamcol_list])  #Tags of beamcols
            beamcol_i_nodeTag = np.array([beamcol.i_nodeTag for beamcol in beamcol_list])  #Tags of i-node of beamcols
            beamcol_j_nodeTag = np.array([beamcol.j_nodeTag for beamcol in beamcol_list])  #Tags of j-node of beamcols
            beamcol_x1 = np.array([beamcol.x1 for beamcol in beamcol_list],dtype=float) #x of inode of beamcols
            beamcol_y1 = np.array([beamcol.y1 for beamcol in beamcol_list],dtype=float) #y of inode of beamcols
            beamcol_z1 = np.array([beamcol.z1 for beamcol in beamcol_list],dtype=float) #z of inode of beamcols
            beamcol_x2 = np.array([beamcol.x2 for beamcol in beamcol_list],dtype=float) #x of jnode of beamcols
            beamcol_y2 = np.array([beamcol.y2 for beamcol in beamcol_list],dtype=float) #y of jnode of beamcols
            beamcol_z2 = np.array([beamcol.z2 for beamcol in beamcol_list],dtype=float) #z of jnode of beamcols
            beamcol_E = np.array([beamcol.E for beamcol in beamcol_list],dtype=float) #E of beamcols
            beamcol_G = np.array([beamcol.G for beamcol in beamcol_list],dtype=float) #G of beamcols
            beamcol_Iy = np.array([beamcol.Iy for beamcol in beamcol_list],dtype=float) #Iy of beamcols
            beamcol_Iz = np.array([beamcol.Iz for beamcol in beamcol_list],dtype=float) #Iy of beamcols
            beamcol_J = np.array([beamcol.J for beamcol in beamcol_list],dtype=float) #J of beamcols
            beamcol_A = np.array([beamcol.A for beamcol in beamcol_list],dtype=float) #A of beamcols

            #Create 'BeamCol' object from arrays of attributes
            beamcols_replica = BeamCol(beamcol_eleTag, beamcol_i_nodeTag, beamcol_j_nodeTag, 
                                                beamcol_x1,beamcol_y1,beamcol_z1,beamcol_x2,beamcol_y2,
                                                beamcol_z2,beamcol_E,beamcol_G,beamcol_Iy,beamcol_Iz,beamcol_J,beamcol_A)

            # Call the functions defined for 'BeamCol' objects
            # Implement jax.vmap and jax.jit to boost the calculation
            # Note that the first call of jax.jit usually takes long because it is "compiling" the codes for future fast runs
            # the following calls will be extremely fast
            
            beamcol_SensKCoord = jnp.array(vmap(BeamColSens.Ele_Sens_K_Coord)(beamcols_replica))  #Get the sensitivity, shape of (6,n_beamcol,12,12)
            
            #From dkdx to dcdx
            dcdx_bc = dcdx_beamcol(beamcol_i_nodeTag,beamcol_j_nodeTag,beamcol_SensKCoord,u)
            
            #Add to the global array
            dcdx_all += dcdx_bc

        return dcdx_all


#External functions

def dcdx_beamcol_expanded(i,j,dkdx,u):
    '''
    Return an ndarray of shape (n_beamcol,3*n_node), representing the sensitivity of the 
    strain energy contributed by each beam-column.
    Each row represents the dcdx contribution from each beamcolumn.
    This function is now written for one beam-column but will be 'vmapped' later to be applied to
    all the beam columns in the system.
    The following inputs and returns are for the 'vmapped' function.

    Inputs
    -----
    i: ndarray of shape (n_beamcol)
        i-node of each beamcolumn,

    j: ndarray of shape (n_beamcol)
        j-node of each beamcolumn

    dkdx: ndarray of shape (6,n_beamcol,12,12)
        sensitivity of local stiffness wrt to nodal coordinates of each beam-column

    u: ndarray of shape (6*n_node)
        displacement vector in global coordinate system

    Returns
    -----
    dcdx_g: ndarray of shape (n_beamcol,3*n_node)
        sensitivity of the strain energy contributed by each beam-column.

    '''

    index_i_node = jnp.linspace(i*6,i*6+5,6,dtype=int) #index of i-node
    index_j_node = jnp.linspace(j*6,j*6+5,6,dtype=int) #index of j-node
    index_beamcol = jnp.hstack((index_i_node,index_j_node)) #stack 'em
    u_e = jnp.asarray(u,dtype=float)[index_beamcol] #displacement vector of this beamcolumn
    dcdx_e = -0.5*u_e.T@dkdx@u_e #adjoint method for sensitivity
    n_node = u.shape[0]/6
    dcdx_g = jnp.zeros(int(3*n_node)) #extened container
    index_i_crd = jnp.linspace(i*3,i*3+2,3,dtype=int) #index for coordinate
    index_j_crd = jnp.linspace(j*3,j*3+2,3,dtype=int) #index for coordiante
    index_crd = jnp.hstack((index_i_crd,index_j_crd)) #stack 'em
    dcdx_g = dcdx_g.at[index_crd].set(dcdx_e) #update the array
    return dcdx_g

@jit
def dcdx_beamcol(i_s,j_s,dkdx,u):
    '''
    Sensitivity of the strain energy wrt nodal coordinates contributed by beam-column elements.

    Inputs
    -----
    i_s: ndarray of shape (n_beamcol)
        i-node of each beamcolumn,

    j_s: ndarray of shape (n_beamcol)
        j-node of each beamcolumn

    dkdx: ndarray of shape (6,n_beamcol,12,12)
        sensitivity of local stiffness wrt to nodal coordinates of each beam-column

    u: ndarray of shape (6*n_node)
        displacement vector in global coordinate system

    Returns
    -----
    ndarray of shape (3*n_node)
        sensitivity of the strain energy wrt nodal coordinates contributed by beam-column.
    '''
    dcdx_ex = vmap(dcdx_beamcol_expanded,(0,0,1,None),0)(i_s,j_s,dkdx,u)
    return jnp.sum(dcdx_ex,axis=0) 