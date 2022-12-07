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


    def delete_node(self, nodeTag):
        '''
        Removes a node from the model and delete elements that are made up from this node.
        
        Inputs
        ----------
        nodeTag : int
            The nodeTag of the node to be removed.
        '''
        
        # Remove the node.
        self.nodes.pop(nodeTag)
        
        # Find any elements attached to the node and remove them
        self.beamcols = {eleTag: beamcol for eleTag, beamcol in self.beamcols.items() if beamcol.i_nodeTag != nodeTag and beamcol.j_nodeTag != nodeTag}
        

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

    
    def Sens_K_Coord(self, sparse=True):
        '''
        1. Assemble the global stiffness matrix 'K'. (Without the b.c.)
        2. Calculate the sensitivity of 'K' w.r.t. the nodal coordinates.


        Inputs
        -----
        sparse : bool
            Indicates whether the sparse matrix is used. Default is True

        Returns:
        ``````
        jac_K_coord: ndarray
            If sparse == True: 
                shape of [n_nodes, 3]
                each entry is a scipy.sparse.coo_matrix represents an ndarray of shape [6*n_nodes, 6*n_nodes]
            If sparse == False: 
                shape of [n_nodes, 3, 6*n_nodes, 6*n_nodes]
        '''

        #Creating a container to store the sensitivity
        if sparse == True:
            from scipy.sparse import coo_matrix # Import `scipy` package if the sparse == True
            
            #Array storing the sensitivity of K w.r.t. nodal coordinates; 
            #each entry is a coo_matrix sparse matrix with a size of (6*num_of_node,6*num_of_node)
            jac_K_Coord = np.zeros((len(self.nodes),3),dtype=object) 

            #Create lists for the coo_matrix
            row = [[] for _ in range(len(self.nodes))] 
            col = [[] for _ in range(len(self.nodes))]
            value_x = [[] for _ in range(len(self.nodes))] 
            value_y = [[] for _ in range(len(self.nodes))] 
            value_z = [[] for _ in range(len(self.nodes))]

            

        else:
            jac_K_Coord = np.zeros((len(self.nodes),3,len(self.nodes)*6,len(self.nodes)*6)) 

        # To implement jax.jit and jax.vmap to boost the calculation of sensitivities
        # and to avoid for-loops, the following codes create several ndarrays consisting of 
        # the attributes of 'BeamCol' objects.
        # We then create a 'BeamCol' object made from attributes arrays [struct of arrays]
        # rather than creating an array storing our 'BeamCol' objects [array of structs]
        # Referred to : https://github.com/google/jax/discussions/5322, answer by shoyer

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
        beamcol_SensKCoord = vmap(BeamColSens.Ele_Sens_K_Coord)(beamcols_replica)  #Get the sensitivity, shape of (6,n_beamcol,12,12)
        beamcol_SensKCoord = np.array(beamcol_SensKCoord) #Convert to np.array, jnp.array is traceable so it is rather slow in for-loops


        # Assemble the element's sensitivity to global jac_K_Coord,
        # step through each beam-column
        i_beamcol = 0 
        for beamcol in self.beamcols.values():
            
            # Step through each term in the beam-column's stiffness matrix (12,12)
            # 'a' & 'b' are rows and cols in its own stiffness matrix (12,12)
            # 'm' & 'n' are rows and cols in the global stiffness matrix (n_node*6, n_node*6)
            for a in range(12):
            
                # Determine if index 'a' is related to the i-node or j-node
                if a < 6:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = beamcol.i_nodeTag*6 + a
                else:
                    # Find the corresponding index 'm' in the global stiffness matrix
                    m = beamcol.j_nodeTag*6 + (a-6)
                
                for b in range(12):
                
                    # Determine if index 'b' is related to the i-node or j-node
                    if b < 6:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = beamcol.i_nodeTag*6 + b
                    else:
                        # Find the corresponding index 'n' in the global stiffness matrix
                        n = beamcol.j_nodeTag*6 + (b-6)
                
                    # Now that 'm' and 'n' are known, place the term in the global stiffness matrix
                    if sparse == True:
                        row[beamcol.i_nodeTag].append(m)
                        row[beamcol.j_nodeTag].append(m)
                        col[beamcol.i_nodeTag].append(n)
                        col[beamcol.j_nodeTag].append(n)
                        value_x[beamcol.i_nodeTag].append(beamcol_SensKCoord[0,i_beamcol,a, b])
                        value_y[beamcol.i_nodeTag].append(beamcol_SensKCoord[1,i_beamcol,a, b])
                        value_z[beamcol.i_nodeTag].append(beamcol_SensKCoord[2,i_beamcol,a, b])
                        value_x[beamcol.j_nodeTag].append(beamcol_SensKCoord[3,i_beamcol,a, b])
                        value_y[beamcol.j_nodeTag].append(beamcol_SensKCoord[4,i_beamcol,a, b])
                        value_z[beamcol.j_nodeTag].append(beamcol_SensKCoord[5,i_beamcol,a, b])
                    else:
                        jac_K_Coord[beamcol.i_nodeTag,0,m,n] += beamcol_SensKCoord[0,i_beamcol,a, b]
                        jac_K_Coord[beamcol.i_nodeTag,1,m,n] += beamcol_SensKCoord[1,i_beamcol,a, b]
                        jac_K_Coord[beamcol.i_nodeTag,2,m,n] += beamcol_SensKCoord[2,i_beamcol,a, b]
                        jac_K_Coord[beamcol.j_nodeTag,0,m,n] += beamcol_SensKCoord[3,i_beamcol,a, b]
                        jac_K_Coord[beamcol.j_nodeTag,1,m,n] += beamcol_SensKCoord[4,i_beamcol,a, b]
                        jac_K_Coord[beamcol.j_nodeTag,2,m,n] += beamcol_SensKCoord[5,i_beamcol,a, b]


            i_beamcol += 1    
       

        if sparse == True:

            #Step through every node
            for i in range(len(self.nodes)):
                #Each entry is a sparse matrix of corresponding sensitivity
                #x
                jac_K_Coord[i,0] = coo_matrix((value_x[i],(row[i],col[i])),shape=(len(self.nodes)*6, len(self.nodes)*6)) 
                #y
                jac_K_Coord[i,1] = coo_matrix((value_y[i],(row[i],col[i])),shape=(len(self.nodes)*6, len(self.nodes)*6)) 
                #z
                jac_K_Coord[i,2] = coo_matrix((value_z[i],(row[i],col[i])),shape=(len(self.nodes)*6, len(self.nodes)*6)) 
        

        return jac_K_Coord


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

    index_i_node = jnp.linspace(i*6,i*6+6,6,dtype=int) #index of i-node
    index_j_node = jnp.linspace(j*6,j*6+6,6,dtype=int) #index of j-node
    index_beamcol = jnp.hstack((index_i_node,index_j_node)) #stack 'em
    u_e = jnp.asarray(u)[index_beamcol] #displacement vector of this beamcolumn
    dcdx_e = u_e.T@dkdx@u_e #adjoint method for sensitivity
    n_node = u.shape[0]/6
    dcdx_g = jnp.zeros(int(3*n_node)) #extened container
    index_i_crd = jnp.linspace(i*3,i*3+3,3,dtype=int) #index for coordinate
    index_j_crd = jnp.linspace(j*3,j*3+3,3,dtype=int) #index for coordiante
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