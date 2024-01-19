'''
A module that helps to build the model for FEA.
Functions for: adding nodes, adding supports, adding loads, etc.
'''
#%%
import numpy as np
import jax.numpy as jnp
from .element import BeamCol,Truss,Quad
from . import solver,mechanics
import jax
jax.config.update("jax_enable_x64", True)


class Node():
    """
    Create a node.

    Attributes
    -----
    nodeTag: int 
        the tag/index of this node
    X, Y, Z: float
        the coordinates of thie node

    """
    
    def __init__(self, nodeTag, X, Y, Z):
        self.nodeTag = nodeTag      # Index of this node
        self.X = X            # Global X coordinate
        self.Y = Y            # Global Y coordinate
        self.Z = Z            # Global Z coordinate

class Model():
    '''
    The FEA model yet to be analyzed.
    '''

    def __init__(self):
        """
        Initialize a 'Model' object.

        """
        
        self.nodes = {} #dictionary for the nodes: key is nodeTag, value is a list of [nodes_x,nodes_y,nodes_z]
        self.beamcols = {}    # Beam-columns in the system
        self.trusses = {}    # Trusses in the system
        self.quads = {} #MITC-4 quad shell elements
        self.elements = {} #including all the elements
        self.known_indices = [] #Storing the indices of the known dofs
        self.f = {} #nodal loads
        self.u = None # FE results


    
    def add_node(self, nodeTag:int, X:int, Y:int, Z:int):
        '''
        Add a node to the model.

        Parameters:
        -----
        nodeTag: int 
            the tag/index of this node

        X, Y, Z: float
            the coordinates of thie node

        '''
        this_node = Node(nodeTag,X,Y,Z)
        # Add the new node/updating an existing node to model
        self.nodes[nodeTag] = this_node

    def update_node(self, nodeTag:int, X:int, Y:int, Z:int):
        '''
        Update the nodal coordinates.

        Parameters:
        -----
        nodeTag: int 
            the tag/index of this node

        X, Y, Z: float
            the coordinates of thie node

        '''
        this_node = Node(nodeTag,X,Y,Z)
        # Updating an existing node to model
        self.nodes[nodeTag] = this_node

    def add_beamcol(self, eleTag:int, i_nodeTag:int, j_nodeTag:int, E:float, G:float,
                     Iy:float, Iz:float, J:float, A:float):
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

    def add_truss(self, eleTag:int, i_nodeTag:int, j_nodeTag:int, E:float, A:float):
        '''
        Adding/updating a truss element to the model
        
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

        A: float
            Area of cross section
        '''
        node_i = self.nodes[i_nodeTag] #first node
        node_j = self.nodes[j_nodeTag] #second node
        ele_crds = [node_i.X,node_i.Y,node_i.Z,node_j.X,node_j.Y,node_j.Z] #coordinates

        #Create/update a beam-column and save it to the dictionary
        new_truss = Truss(eleTag, node_i.nodeTag, node_j.nodeTag, ele_crds, E, A)
        self.trusses[eleTag] = new_truss
    
    def add_quad(self, eleTag:int, i_nodeTag:int, j_nodeTag:int, m_nodeTag:int, n_nodeTag:int, 
                t:float, E:float, nu:float, kx_mod=1.0, ky_mod=1.0):
        '''
        Adding/updating a quad shell element to the model
        
        Inputs
        ----------
        eleTag : int
            Index of this element

        i_nodeTag,j_nodeTag,m_nodeTag,n_nodeTag : int
            The tag of the node of the quad

        t: float
            Thickness of the quad

        E: float
            Young's modulus
        
        nu: float
            Poisson's ratio
        
        kx_mod, ky_mod: float
            Stiffness modification factor
        '''
        node_i = self.nodes[i_nodeTag] #first node
        node_j = self.nodes[j_nodeTag] #second node
        node_m = self.nodes[m_nodeTag] #third node
        node_n = self.nodes[n_nodeTag] #fourth node
        
        ele_crds = [node_i.X,node_i.Y,node_i.Z,node_j.X,node_j.Y,node_j.Z,
                    node_m.X,node_m.Y,node_m.Z,node_n.X,node_n.Y,node_n.Z] #coordinates

        #Create/update a beam-column and save it to the dictionary
        new_quad = Quad(eleTag, i_nodeTag, j_nodeTag, m_nodeTag, n_nodeTag, 
                ele_crds, t, E, nu, kx_mod, ky_mod)
        self.quads[eleTag] = new_quad
    
    def add_support(self,nodeTag,active_supports=[1,1,1,1,1,1]):
        '''
        Adding nodal support in the model.
        
        Parameters
        -----
        nodeTag: int
            The index of the node.
        
        active_supports: list, 1darray
            Whether the support for [X,Y,Z,RX,RY,RZ] is active.
            1 is active, 0 is deactivated.
            Default is fixed.
        '''
        dofs = np.linspace(nodeTag*6,nodeTag*6+5,6,dtype='int32')
        active_supports = np.array(active_supports,dtype='int32') #convert to standard np.array
        where_active = np.argwhere(active_supports==1).ravel() #which dofs are active
        self.known_indices.extend(dofs[where_active]) #append to the global list

    def add_nodal_load(self,nodeTag,nodal_load=[0.0,0.0,0.0,0.0,0.0,0.0]):
        '''
        Adding a nodal load to a node.

        Parameters
        -----
        nodeTag: int
            The index of the node.
        
        nodal_load: list
            The applied nodal load in the order of [fx,fy,fz,mx,my,mz]
        '''
        self.f[nodeTag] = nodal_load
    
    # A function called model_freeze.
    # Which updates some attributes of the model, such as node_crds, loads, etc.

    def model_ready(self):
        '''
        Update attributes of the model, such as node_crds, loads, etc. Essential before conducting FEA or optimization.
        '''
        #node coordinates
        self.crds = self.get_node_crds()

        #Nodal loads
        self.nodal_loads = jnp.array(self.get_loads()) #jnp.conversion needed

        #Boundary condition ids
        self.known_id,self.unknown_id = self.get_boundary_ids()

        #Dof
        self.ndof = self.get_dofs()

        #Beamcols
        self.n_beamcol = len(self.beamcols)
        self.cnct_beamcols = self.get_cnct_beamcols()
        self.prop_beamcols = self.get_beamcols_cross_prop()

        #Quads
        self.n_quad = len(self.quads)
        self.cnct_quads = self.get_cnct_quads()
        self.prop_quads = self.get_quads_cross_prop()

    def get_node_crds(self):
        '''
        Get the coordinates of all nodes.
        Return a 2D array of shape (n_node,3)
        '''
        xs = jnp.array([[nd.X for nd in self.nodes.values()]],dtype=float)
        ys = jnp.array([[nd.Y for nd in self.nodes.values()]],dtype=float)
        zs = jnp.array([[nd.Z for nd in self.nodes.values()]],dtype=float)
        return jnp.vstack((xs,ys,zs)).T
    
    def get_loads(self):
        '''
        Get the load vector in 1D.
        Return a 1D array of shape (n_node*6)

        Note: get rid of the for loop.
        '''
        dof = self.get_dofs()
        load = np.zeros(dof)
        for node in self.f.keys():
            load[node*6:node*6+6] = self.f[node] 

        return load

    def get_boundary_ids(self):
        '''
        Return the indices of unknown and known dof-displacement based on the boundary conditions.
        
        Returns
        ------
        known_indices: ndarray 
            indices of displacement that are known. 
        
        unknown_indices: ndarray
            indices of unknown displacement
        '''
        known_id = (np.array(self.known_indices,dtype='int32')).ravel() #convert the active support indices list to np.array
        all_indices = np.linspace(0,6*len(self.nodes)-1,6*len(self.nodes),dtype='int32') #create a container for all indices
        unknown_id = all_indices[np.where(np.isin(all_indices,known_id,assume_unique=True,invert=True))] #slice out the known indices 
        return known_id,unknown_id

    def get_dofs(self):
        '''
        Get the dofs of the system.
        '''
        return len(self.nodes.values())*6
    
    def get_cnct_beamcols(self):
        '''
        Get the connectivity matrix of beam-columns.
        Return a 2D array of shape (n_beamcol,2)
        '''
        i_nodes_tags = np.array([[bc.i_nodeTag for bc in self.beamcols.values()]],dtype='int32')
        j_nodes_tags = np.array([[bc.j_nodeTag for bc in self.beamcols.values()]],dtype='int32')
        return np.vstack((i_nodes_tags,j_nodes_tags)).T
    
    def get_cnct_quads(self):
        '''
        Get the connectivity matrix of quad elements.
        Return a 2D array of shape (n_quad,4)
        '''
        i_nodes_tags = np.array([[qd.i_nodeTag for qd in self.quads.values()]],dtype='int32')
        j_nodes_tags = np.array([[qd.j_nodeTag for qd in self.quads.values()]],dtype='int32')
        m_nodes_tags = np.array([[qd.m_nodeTag for qd in self.quads.values()]],dtype='int32')
        n_nodes_tags = np.array([[qd.n_nodeTag for qd in self.quads.values()]],dtype='int32')
        return np.vstack((i_nodes_tags,j_nodes_tags,m_nodes_tags,n_nodes_tags)).T
    
    def get_beamcols_cross_prop(self):
        '''
        Get cross-sectional properties: E, G, Iy, Iz, J, A
        Return a 2D array of shape (n_beamcol,6)
        '''
        Es = np.array([[bc.E for bc in self.beamcols.values()]])
        Gs = np.array([[bc.G for bc in self.beamcols.values()]])
        Iys = np.array([[bc.Iy for bc in self.beamcols.values()]])
        Izs = np.array([[bc.Iz for bc in self.beamcols.values()]])
        Js = np.array([[bc.J for bc in self.beamcols.values()]])
        As = np.array([[bc.A for bc in self.beamcols.values()]])
        return np.vstack((Es, Gs, Iys, Izs, Js, As)).T
    
    def get_quads_cross_prop(self):
        '''
        Get cross-sectional properties of quads: t, E, nu, kx_mod, ky_mod
        Return a 2D array of shape (n_quad,5)
        '''
        ts = np.array([[qd.t for qd in self.quads.values()]])
        Es = np.array([[qd.E for qd in self.quads.values()]])
        nus = np.array([[qd.nu for qd in self.quads.values()]])
        kx_mods = np.array([[qd.kx_mod for qd in self.quads.values()]])
        ky_mods = np.array([[qd.ky_mod for qd in self.quads.values()]])
        return np.vstack((ts, Es, nus, kx_mods, ky_mods)).T
    
    def solve(self,which_solver='dense',enforce_scipy_sparse = True):
        '''
        Solve the linear system to obtain the displacement vector.
        The solver depends on the devices being used:
            Dense:
                cpu & gpu: jax.numpy's dense solve
            Sparse:
                cpu: original scipy's spsolve TODO:Should be default? Refer to Jax-FDM
                gpu: jax.experimental.sparse.linalg.spsolve TODO: seems unstable, sometimes output "singular matrix" error when fine using dense solver
        
        Parameters:
            which_meter: str, either 'sparse' or 'dense'.
            enforce_scipy_sparse: bool, True if using scipy's sparse solver no matter what device is being used
        '''
        K_aug = mechanics.model_K_aug(self) #LHS
        f_aug = mechanics.model_f_aug(self) #RHS
        ndof = self.get_dofs() #number of dofs in the system
        if which_solver == 'dense':
            self.u = solver.jax_dense_solve(K_aug,f_aug)[:ndof]
        elif which_solver == 'sparse':
            if enforce_scipy_sparse:
                self.u = solver.sci_sparse_solve(K_aug,f_aug)[:ndof]
            else:
                if jax.default_backend() == 'gpu':
                    self.u = solver.jax_sparse_solve(K_aug,f_aug)[:ndof]
                elif jax.default_backend() == 'cpu':
                    self.u = solver.sci_sparse_solve(K_aug,f_aug)[:ndof]
        else:
            print("Please select the right solver: dense or sparse")

