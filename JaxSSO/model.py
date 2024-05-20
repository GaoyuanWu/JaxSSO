'''
A module that helps to build the model for FEA.
Functions for: adding nodes, adding supports, adding loads, etc.
'''
#%%
import numpy as np
import jax.numpy as jnp
from .element import BeamCol,Truss,Quad
from . import solver,assemblemodel
import jax
from jax.tree_util import register_pytree_node_class
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
    The FE model yet to be analyzed.
    '''

    def __init__(self):
        """
        Initialize a 'Model' object.

        """

        self.nodes = {} #dictionary for the nodes: key is nodeTag, value is a list of [nodes_x,nodes_y,nodes_z]
        self.beamcols = {}    # dictionary for the beamcols: key is eleTag, value is a list of beamcol attributes
        #self.trusses = {}    # Trusses in the system, TODO: NOT YET IMPLEMENTED
        self.quads = {} # dictionary for the MITC-4 quad shell elements: key is eleTag, value is a list of beamcol attributes
        #self.elements = {} #including all the elements, TODO
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
        key = nodeTag #key
        attr = [X,Y,Z] #attributes
        self.nodes[key] = attr # Add the new node/updating an existing node to model

    def update_node(self, nodeTag:int, XYZ:int, value:float):
        '''
        Update the nodal coordinates.

        Parameters:
        -----
        nodeTag: int 
            the tag/index of the node you want to modify

        XYZ: int
            take a value from (0,1,2), indicating which coordinate (x or y or z) to modify
        
        value: float
            new value

        '''
        # Updating an existing node to model
        try:
            self.nodes[nodeTag][XYZ] = value
        except:
            print("Node {} does not exist in the model".format(nodeTag))

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
        #Create/update a beam-column and save it to the dictionary
        new_beamcol = BeamCol(eleTag, i_nodeTag, j_nodeTag, E, G, Iy, Iz, J, A)
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
        pass
    
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
        #Create/update a beam-column and save it to the dictionary
        new_quad = Quad(eleTag, i_nodeTag, j_nodeTag, m_nodeTag, n_nodeTag, 
                t, E, nu, kx_mod, ky_mod)
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
    
    #-------------------------------------------------------------------------------
    # Warm up the model using 'model_ready' to 'freeze' the model ready for analysis
    # Will be fed to/modified by 'SSO_model' if doing optimization
    #-------------------------------------------------------------------------------
    
    def model_ready(self):
        '''
        Update attributes of the model, such as node_crds, loads, etc. Essential before conducting FEA or optimization.
        '''
                
        #Nodal coordinates
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
        self.prop_beamcols = jnp.array(self.get_beamcols_cross_prop())

        #Quads
        self.n_quad = len(self.quads)
        self.cnct_quads = self.get_cnct_quads()
        self.prop_quads = jnp.array(self.get_quads_cross_prop())        

    def get_node_crds(self):
        '''
        Get the coordinates of all nodes.
        Return a 2D array of shape (n_node,3)
        '''
        xs = jnp.array([[xyz[0] for xyz in self.nodes.values()]],dtype=float)
        ys = jnp.array([[xyz[1] for xyz in self.nodes.values()]],dtype=float)
        zs = jnp.array([[xyz[2] for xyz in self.nodes.values()]],dtype=float)
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
    
    def select_solver(self,which_solver,enforce_scipy_sparse):
        '''
        Determine which solver to use
        '''
        if which_solver == 'dense':
            return solver.jax_dense_solve
        elif which_solver == 'sparse':
            if enforce_scipy_sparse:
                return solver.sci_sparse_solve
            else:
                if jax.default_backend() == 'gpu':
                    return solver.jax_sparse_solve
                elif jax.default_backend() == 'cpu':
                    print("Cannot use JAX's sparse solver because it only supports GPU at the moment")
                    return solver.sci_sparse_solve
        else:
            print("Please select the right solver: dense or sparse")

    def solve(self,which_solver='sparse',enforce_scipy_sparse = True):
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
        K_aug = assemblemodel.model_K_aug(self) #LHS
        f_aug = assemblemodel.model_f_aug(self) #RHS
        ndof = self.get_dofs() #number of dofs in the system
        solver_fea = self.select_solver(which_solver,enforce_scipy_sparse)
        self.u = solver_fea(K_aug,f_aug)[:ndof]
    

    def strain_energy(self):
        if self.u != None:
            return 0.5 * self.nodal_loads @ self.u
        else:
            print("Model has not been analyzed yet.")