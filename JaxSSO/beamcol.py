'''
Modules for BeamColumn elements.


References
---------
1. Bathe, K. J. (2006). Finite element procedures. Klaus-Jurgen Bathe.
2. "A First Course in the Finite Element Method, 4th Edition", Daryl L. Logan
3. "Finite Element Analysis Fundamentals", Richard H. Gallagher
4. PyNite: An open-source FEA solver for python; https://github.com/JWock82/PyNite
'''
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import  jit,vmap
from jax.experimental import sparse
import jax
jax.config.update("jax_enable_x64", True)

#%%
class BeamCol():
    '''
    A class for a beam-column element.

    Parameters
    -----
    eleTag: int
        The tag of this element

    i_nodeTag, j_nodeTag: int
        The tags of the i-node and j-node

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
        
    def __init__(self, eleTag, i_nodeTag, j_nodeTag, E, G, Iy, Iz, J, A):
        
        #Inputs, Children of Pytree
        self.dofs = 2 * 6 #number of dofs of this element
        self.eleTag = eleTag    # Element tag
        self.i_nodeTag = i_nodeTag # i-node tag
        self.j_nodeTag = j_nodeTag
        self.E = E  # The modulus of elasticity of the element
        self.G = G  # The shear modulus of the element
        self.Iy = Iy  # The y-axis moment of inertia
        self.Iz = Iz  # The z-axis moment of inertia
        self.J = J  # The torsional constant
        self.A = A  # The cross-sectional area
    
    @staticmethod
    def T(crds):
        '''
        Returns the transformation matrix between local and global axis.
        Note the following convention is used:
            Local x: longitudinal
            Local y: local x-y plane parallel to global z, local y pointing to global +z
            Local z: horizontal
        Reference: http://www.gcg.ufjf.br/pub/doc132.pdf and https://www.engissol.com/Downloads/Technical%20Notes%20and%20examples.pdf
        '''
        x1,y1,z1,x2,y2,z2 = crds
        length = jnp.sqrt((x1 - x2)**2+(y1-y2)**2+(z1 - z2)**2) #the length
        zero_entry = 0
        one_entry = 1
    

        #Projections of C 
        Cxz_vec = jnp.array([(x2-x1)/length, zero_entry, (z2-z1)/length]) 
        Cxz = jnp.linalg.norm(Cxz_vec,axis=0)
        Cx = (x2-x1)/length #projection on x
        Cy = (y2-y1)/length #projection on y
        Cz = (z2-z1)/length #projection on z
        sin_alpha = zero_entry
        cos_alpha = one_entry

        #Check if the member only has y-difference
        dirCos = jnp.where(Cxz==0,jnp.array([[zero_entry, Cy, zero_entry],
                           [-Cy*cos_alpha,  zero_entry, -cos_alpha],
                           [-Cy*cos_alpha, zero_entry, sin_alpha]]),
                      jnp.array([[Cx,Cy,Cz],
                                 [-1*(Cx*Cy*sin_alpha-Cz*cos_alpha)/Cxz, Cxz*sin_alpha, -1*(Cy*Cz*sin_alpha+Cx*cos_alpha)/Cxz ],
                                 [(-Cx*Cy*cos_alpha-Cz*sin_alpha)/Cxz, Cxz*cos_alpha,(-Cy*Cz*cos_alpha+Cx*sin_alpha)/Cxz]]))

        # Build the transformation matrix
        transMatrix = jnp.zeros((12, 12))
        transMatrix = transMatrix.at[0:3,0:3].set(dirCos)
        transMatrix = transMatrix.at[3:6, 3:6].set(dirCos)
        transMatrix = transMatrix.at[6:9, 6:9].set(dirCos)
        transMatrix = transMatrix.at[9:12, 9:12].set(dirCos)
        return transMatrix
    
    @staticmethod
    def K_local(crds,E,G,Iy,Iz,J,A):
        '''
        Return the element's stiffness matrix WITHOUT coordinate transformation
        '''
        x1,y1,z1,x2,y2,z2 = crds
        L = jnp.sqrt((x1 - x2)**2+(y1-y2)**2+(z1 - z2)**2) #length
        zeros_entry = 0 #zero entry in the matrix
        k = jnp.array([[A*E/L,  zeros_entry,             zeros_entry,             zeros_entry,      zeros_entry,            zeros_entry,            -A*E/L, zeros_entry,             zeros_entry,             zeros_entry,      zeros_entry,            zeros_entry],
                        [zeros_entry,      12*E*Iz/L**3,  zeros_entry,             zeros_entry,      zeros_entry,            6*E*Iz/L**2,  zeros_entry,      -12*E*Iz/L**3, zeros_entry,             zeros_entry,      zeros_entry,            6*E*Iz/L**2],
                        [zeros_entry,      zeros_entry,             12*E*Iy/L**3,  zeros_entry,      -6*E*Iy/L**2, zeros_entry,            zeros_entry,      zeros_entry,             -12*E*Iy/L**3, zeros_entry,      -6*E*Iy/L**2, zeros_entry],
                        [zeros_entry,      zeros_entry,             zeros_entry,             G*J/L,  zeros_entry,            zeros_entry,            zeros_entry,      zeros_entry,             zeros_entry,             -G*J/L, zeros_entry,            zeros_entry],
                        [zeros_entry,      zeros_entry,             -6*E*Iy/L**2,  zeros_entry,      4*E*Iy/L,     zeros_entry,            zeros_entry,      zeros_entry,             6*E*Iy/L**2,   zeros_entry,      2*E*Iy/L,     zeros_entry],
                        [zeros_entry,      6*E*Iz/L**2,   zeros_entry,             zeros_entry,      zeros_entry,            4*E*Iz/L,     zeros_entry,      -6*E*Iz/L**2,  zeros_entry,             zeros_entry,      zeros_entry,            2*E*Iz/L],
                        [-A*E/L, zeros_entry,             zeros_entry,             zeros_entry,      zeros_entry,            zeros_entry,            A*E/L,  zeros_entry,             zeros_entry,             zeros_entry,      zeros_entry,            zeros_entry],
                        [zeros_entry,      -12*E*Iz/L**3, zeros_entry,             zeros_entry,      zeros_entry,            -6*E*Iz/L**2, zeros_entry,      12*E*Iz/L**3,  zeros_entry,             zeros_entry,      zeros_entry,            -6*E*Iz/L**2],
                        [zeros_entry,      zeros_entry,             -12*E*Iy/L**3, zeros_entry,      6*E*Iy/L**2,  zeros_entry,            zeros_entry,      zeros_entry,             12*E*Iy/L**3,  zeros_entry,      6*E*Iy/L**2,  zeros_entry],
                        [zeros_entry,      zeros_entry,             zeros_entry,             -G*J/L, zeros_entry,            zeros_entry,            zeros_entry,      zeros_entry,             zeros_entry,             G*J/L,  zeros_entry,            zeros_entry],
                        [zeros_entry,      zeros_entry,             -6*E*Iy/L**2,  zeros_entry,      2*E*Iy/L,     zeros_entry,            zeros_entry,      zeros_entry,             6*E*Iy/L**2,   zeros_entry,      4*E*Iy/L,     zeros_entry],
                        [zeros_entry,      6*E*Iz/L**2,   zeros_entry,             zeros_entry,      zeros_entry,            2*E*Iz/L,     zeros_entry,      -6*E*Iz/L**2,  zeros_entry,             zeros_entry,      zeros_entry,            4*E*Iz/L]])
    
        return k
    
    @staticmethod
    def element_K_beamcol(crds,E,G,Iy,Iz,J,A):
        '''
        Given the attributes of a beam-column. Return element's stiffness matrix in global coordinate system.
        The element's stiffness matrix is flattened (raveled).
        '''
        ele_K_local = BeamCol.K_local(crds,E,G,Iy,Iz,J,A) # Element's stiffness matrix, before coordinate trnasformation
        ele_T = BeamCol.T(crds) #Coordinate transormation matrix
        ele_K_global = jnp.matmul(jnp.linalg.solve(ele_T,ele_K_local),ele_T)
        return ele_K_global
    
    @staticmethod
    def element_K_beamcol_indices(i_nodeTag,j_nodeTag):
        '''
        Given the node tags of a beam-column, return the corresponding indices (rows, columns) of this beam column in the global stiffness matrix.
        '''
        indices_dof = jnp.hstack((jnp.linspace(i_nodeTag*6,i_nodeTag*6+5,6,dtype='int32'),jnp.linspace(j_nodeTag*6,j_nodeTag*6+5,6,dtype='int32'))) #indices represented the dofs of this beamcol
        rows,cols = jnp.meshgrid(indices_dof,indices_dof,indexing='ij')#rows and columns
        indices = jnp.vstack((rows.ravel(),cols.ravel())).T #indices in global stiffness matrix
        return indices


    #--------------------------------------------
    # The following methods are 'vmapped' (vectorized)
    # that are used to output attributes for ALL beamcolumns in the system
    #--------------------------------------------
    
    @staticmethod
    @jit
    def K_local_beamcol(node_crds,prop,cnct):
        '''
        Return ndarray that stores the local stiffness matrix of ALL beam-columns.

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

    @staticmethod
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

    @staticmethod
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

        data = jnp.ravel(vmap(BeamCol.element_K_beamcol)(e_crds,Es,Gs,Iys,Izs,Js,As))  #Get the stiffness values, raveled
        data = data.reshape(-1) # re-dimension the data to 1darray (144*n_beamcol,)
        indices = vmap(BeamCol.element_K_beamcol_indices)(i_nodeTags,j_nodeTags) # Get the indices
        indices = indices.reshape(-1,2) #re-dimension to 2darray of shape (144*n_beamcol,2)
        K= sparse.BCOO((data,indices),shape=(ndof,ndof))
        return K