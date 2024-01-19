'''
Modules including supported elements for Jax_FEA:
    1. Beam-column
    2. Truss
    3. Quadrilateral shell (MITC-4)


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
from jax import  jit
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

    ele_crds: array-like of shape (6,)
        Coordinates of the i-node and j-node: (x_i,y_i,z_i,x_j,y_j,z_j)

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
        
    def __init__(self, eleTag, i_nodeTag, j_nodeTag, crds, E, G, Iy, Iz, J, A):
        
        #Inputs, Children of Pytree
        self.dofs = 2 * 6 #number of dofs of this element
        self.eleTag = eleTag    # Element tag
        self.i_nodeTag = i_nodeTag # i-node tag
        self.j_nodeTag = j_nodeTag
        self.ele_crds = crds #coordinates
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

#%%
class Truss():
    '''
    A class for a truss element.

    Parameters
    -----
    eleTag: int
        The tag of this element

    i_nodeTag, j_nodeTag: int
        The tags of the i-node and j-node

    ele_crds: jnp.array of shape (6,)
        Coordinates of the i-node and j-node: (x_i,y_i,z_i,x_j,y_j,z_j)

    E: float
        Young's modulus

    A: float
        Area of cross section
    '''
        
    def __init__(self, eleTag, i_nodeTag, j_nodeTag, crds, E, A):
        
        #Inputs, Children of Pytree
        self.dofs = 2 * 6 #number of dofs of this element
        self.eleTag = eleTag    # Element tag
        self.i_nodeTag = i_nodeTag # i-node tag
        self.j_nodeTag = j_nodeTag
        self.ele_crds = crds #coordinates
        self.E = E  # The modulus of elasticity of the element
        self.A = A  # The cross-sectional area

    @staticmethod
    def T(crds):
        '''
        Returns the transformation matrix between local and global axis.
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
                           [-Cy*cos_alpha, zero_entry, sin_alpha],
                           [Cy*cos_alpha,  zero_entry, cos_alpha]]),
                      jnp.array([[Cx,Cy,Cz],
                            [(-Cx*Cy*cos_alpha-Cz*sin_alpha)/Cxz, Cxz*cos_alpha,(-Cy*Cz*cos_alpha+Cx*sin_alpha)/Cxz],
                                 [(Cx*Cy*sin_alpha-Cz*cos_alpha)/Cxz, -Cxz*sin_alpha, (Cy*Cz*sin_alpha+Cx*cos_alpha)/Cxz ]]))

        # Build the transformation matrix
        transMatrix = jnp.zeros((12, 12))
        transMatrix = transMatrix.at[0:3,0:3].set(dirCos)
        transMatrix = transMatrix.at[3:6, 3:6].set(dirCos)
        transMatrix = transMatrix.at[6:9, 6:9].set(dirCos)
        transMatrix = transMatrix.at[9:12, 9:12].set(dirCos)
        return transMatrix
    
#%%
class Quad():
    '''
    A class for a quadrilateral shell element based on MITC4 theory.

    Parameters
    -----
    eleTag: int
        The tag of this element

    i_nodeTag, j_nodeTag, m_nodeTag, n_nodeTag: int
        Node tag of i-node and j-node of the element

    crds: 1d array of shape (12,)
        The coordinate of the 4 nodes of this element (xi,yi,zi,xj,yj,zj,xm,ym,zm,xn,yn,zn).
    
    t: float
        Thickness

    E: float
        Young's modulus

    nu: float
        Poisson's ratio

    kx_mod: float
        Stiffness modification factor for local x axis

    ky_mod: float
        Stiffness modification factor for local y axis
    '''
    def __init__(self, eleTag, i_nodeTag, j_nodeTag, m_nodeTag, n_nodeTag, 
                crds, t, E, nu, kx_mod=1.0, ky_mod=1.0):
        
        #Inputs
        self.dofs = 4*6 #numbder of dofs of this element
        self.eleTag = eleTag    # Element tag
        self.i_nodeTag = i_nodeTag #i-node's tag
        self.j_nodeTag = j_nodeTag #j-node's tag
        self.m_nodeTag = m_nodeTag #m-node's tag
        self.n_nodeTag = n_nodeTag #n-node's tag
        self.ele_crds = crds #Coordinates
        self.t = t # The thickness of the element
        self.E = E  # The modulus of elasticity of the element
        self.nu = nu  # The Poisson's ratio
        self.kx_mod = kx_mod  # The stiffness modification
        self.ky_mod = ky_mod  # The stiffness modification
    
    
    @staticmethod 
    def A(crds):
        '''
        Return the surface area of this shell element based on the coordinate
        '''
        #Area of this shell
        X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4 = crds

        edge_12 = jnp.array([X2 - X1, Y2 - Y1, Z2 - Z1]).T
        edge_14 = jnp.array([X4 - X1, Y4 - Y1, Z4 - Z1]).T
        edge_32 = jnp.array([X2 - X3, Y2 - Y3, Z2 - Z3]).T
        edge_34 = jnp.array([X4 - X3, Y4 - Y3, Z4 - Z3]).T
        
        
        area_1 = jnp.linalg.norm(jnp.cross(edge_12, edge_14)) * 0.5
        area_2 = jnp.linalg.norm(jnp.cross(edge_34, edge_32)) * 0.5
        return area_1+area_2
    @staticmethod
    def loc_crds_new(crds):
        ''' 
        Get the nodal coordinates in local coordinate system.
        This local coordinate system is different than coventional local coordiante of quad to avoid non-differentialbility.

        # Non-differentiable when alpha/beta == 0  
        # which means we have to avoid the scenario when r-axis/s-axis is parallel to local x-axis
        # in conventional local coordinate, it is very common.
        # In our cooridinate system, the local 3-1 vector is the local x-axis. which can avoid the
        # aforementioned issue.
        
        '''

        # The vector from node 3 to other nodes
        # Node 3 is the origin of local coordinate
        X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4 = crds

        vector_32 = jnp.array([X2 - X3, Y2 - Y3, Z2 - Z3]).T
        vector_31 = jnp.array([X1 - X3, Y1 - Y3, Z1 - Z3]).T
        vector_34 = jnp.array([X4 - X3, Y4 - Y3, Z4 - Z3]).T
        vector_42 = jnp.array([X2 - X4, Y2 - Y4, Z2 - Z4]).T

        # Define the plate's local x, y, and z axes
        x_axis = vector_31
        z_axis = jnp.cross(x_axis, vector_42)
        z_axis_tri1 = jnp.cross(x_axis,vector_32) #normal based on node 1,2,3
        z_axis_tri2 = jnp.cross(vector_34,x_axis) #normal based on node 1,3,4
        y_axis = jnp.cross(z_axis, x_axis)

        # Convert the x and y axes into unit vectors
        x_axis = x_axis/jnp.linalg.norm(x_axis)
        y_axis = y_axis/jnp.linalg.norm(y_axis)
        z_axis = z_axis/jnp.linalg.norm(z_axis)
        z_axis_tri1 = z_axis_tri1/jnp.linalg.norm(z_axis_tri1) #normal based on node 1,2,3
        z_axis_tri2 = z_axis_tri2/jnp.linalg.norm(z_axis_tri2) #normal based on node 1,3,4

        # Calculate the local (x, y) coordinates for each node
        # Node 3 of this element is the origin
        x1_loc = jnp.dot(vector_31, x_axis)
        x2_loc = jnp.dot(vector_32, x_axis)
        x3_loc = 0.
        x4_loc = jnp.dot(vector_34, x_axis)
        y1_loc = jnp.dot(vector_31, y_axis)
        y2_loc = jnp.dot(vector_32, y_axis)
        y3_loc = 0.
        y4_loc = jnp.dot(vector_34, y_axis)
        new_xys = jnp.array([x1_loc,y1_loc,x2_loc,y2_loc,x3_loc,y3_loc,x4_loc,y4_loc]) #New local x and y
        normal = z_axis
        normal_tri1 = z_axis_tri1
        normal_tri2 = z_axis_tri2
        return new_xys,normal,normal_tri1,normal_tri2
    
    @staticmethod
    def loc_crds(crds):
        ''' 
        Get the nodal coordinates in local coordinate system.
        This local coordinate system is different than coventional local coordiante of quad to avoid non-differentialbility.

        # Non-differentiable when alpha/beta == 0  
        # which means we have to avoid the scenario when r-axis/s-axis is parallel to local x-axis
        # in conventional local coordinate, it is very common.
        # In our cooridinate system, the local 3-1 vector is the local x-axis. which can avoid the
        # aforementioned issue.
        
        '''

        # The vector from node 3 to other nodes
        X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4 = crds

        # Vectors
        vector_32 = jnp.array([X2 - X3, Y2 - Y3, Z2 - Z3]).T
        vector_31 = jnp.array([X1 - X3, Y1 - Y3, Z1 - Z3]).T
        vector_34 = jnp.array([X4 - X3, Y4 - Y3, Z4 - Z3]).T
        vector_42 = jnp.array([X2 - X4, Y2 - Y4, Z2 - Z4]).T

        # Define the plate's local x, y, and z axes
        x_axis = vector_34
        z_axis_tri1 = jnp.cross(vector_31,vector_32) #normal based on node 1,2,3
        z_axis_tri2 = jnp.cross(x_axis,vector_31) #normal based on node 1,3,4
        z_axis = jnp.cross(x_axis, vector_32)
        y_axis = jnp.cross(z_axis, x_axis)

        # Convert the x and y axes into unit vectors
        x_axis = x_axis/jnp.linalg.norm(x_axis)
        y_axis = y_axis/jnp.linalg.norm(y_axis)
        z_axis = z_axis/jnp.linalg.norm(z_axis)
        z_axis_tri1 = z_axis_tri1/jnp.linalg.norm(z_axis_tri1) #normal based on node 1,2,3
        z_axis_tri2 = z_axis_tri2/jnp.linalg.norm(z_axis_tri2) #normal based on node 1,3,4

        # Calculate the local (x, y) coordinates for each node
        # Node 3 of this element is the origin
        x1_loc = jnp.dot(vector_31, x_axis)
        x2_loc = jnp.dot(vector_32, x_axis)
        x3_loc = 0.
        x4_loc = jnp.dot(vector_34, x_axis)
        y1_loc = jnp.dot(vector_31, y_axis)
        y2_loc = jnp.dot(vector_32, y_axis)
        y3_loc = 0.
        y4_loc = jnp.dot(vector_34, y_axis)
        new_xys = jnp.array([x1_loc,y1_loc,x2_loc,y2_loc,x3_loc,y3_loc,x4_loc,y4_loc]) #New local x and y
        normal = z_axis
        normal_tri1 = z_axis_tri1
        normal_tri2 = z_axis_tri2
        return new_xys,normal,normal_tri1,normal_tri2
    @staticmethod
    def T(crds):
        '''
        Returns the coordinate transformation matrix for the quad element.
        This is the conventional coordinate system that may cause non-differentiability problems.
        '''

        # The vector from node 3 to other nodes
        X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4 = crds

        # Vectors
        vector_32 = jnp.array([X2 - X3, Y2 - Y3, Z2 - Z3]).T
        vector_31 = jnp.array([X1 - X3, Y1 - Y3, Z1 - Z3]).T
        vector_34 = jnp.array([X4 - X3, Y4 - Y3, Z4 - Z3]).T
        vector_42 = jnp.array([X2 - X4, Y2 - Y4, Z2 - Z4]).T

        # Define the plate's local x, y, and z axes
        x_axis = vector_34
        z_axis = jnp.cross(x_axis, vector_32)
        y_axis = jnp.cross(z_axis, x_axis)

        # Convert the x and y axes into unit vectors
        x_axis = x_axis/jnp.linalg.norm(x_axis)
        y_axis = y_axis/jnp.linalg.norm(y_axis)
        z_axis = z_axis/jnp.linalg.norm(z_axis)

        # Create the direction cosines matrix.
        dirCos = jnp.array([x_axis,
                        y_axis,
                        z_axis])
        
        # Build the transformation matrix.
        T = jnp.zeros((24, 24))

        # Create arrays for the indices
        T_rows = np.array([ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,
        5,  6,  6,  6,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10, 10, 10, 11,
       11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
       17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22,
       22, 23, 23, 23],dtype='int32') #Row indices
        
        T_cols = np.array([ 0,  1,  2,  0,  1,  2,  0,  1,  2,  3,  4,  5,  3,  4,  5,  3,  4,
        5,  6,  7,  8,  6,  7,  8,  6,  7,  8,  9, 10, 11,  9, 10, 11,  9,
       10, 11, 12, 13, 14, 12, 13, 14, 12, 13, 14, 15, 16, 17, 15, 16, 17,
       15, 16, 17, 18, 19, 20, 18, 19, 20, 18, 19, 20, 21, 22, 23, 21, 22,
       23, 21, 22, 23],dtype='int32') #Column indices

        # flatten and then extend dirCos
        ex_dirCos = jnp.tile(dirCos.reshape(-1),8)

        # assign values to T compactly 
        T = T.at[T_rows,T_cols].set(ex_dirCos)
        
        # Return the transformation matrix.
        return T
      
    @staticmethod
    def T_new(crds):
        '''
        Returns the coordinate transformation matrix for the quad element.
        '''

        # The vector from node 3 to other nodes
        X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4 = crds

        # Vectors
        vector_32 = jnp.array([X2 - X3, Y2 - Y3, Z2 - Z3]).T
        vector_31 = jnp.array([X1 - X3, Y1 - Y3, Z1 - Z3]).T
        vector_34 = jnp.array([X4 - X3, Y4 - Y3, Z4 - Z3]).T
        vector_42 = jnp.array([X2 - X4, Y2 - Y4, Z2 - Z4]).T

        # Define the plate's local x, y, and z axes
        x_axis = vector_31
        z_axis = jnp.cross(x_axis, vector_42)
        y_axis = jnp.cross(z_axis, x_axis)

        # Convert the x and y axes into unit vectors
        x_axis = x_axis/jnp.linalg.norm(x_axis)
        y_axis = y_axis/jnp.linalg.norm(y_axis)
        z_axis = z_axis/jnp.linalg.norm(z_axis)

        # Create the direction cosines matrix.
        dirCos = jnp.array([x_axis,
                        y_axis,
                        z_axis])
        
        # Build the transformation matrix.
        T = jnp.zeros((24, 24))

        # Create arrays for the indices
        T_rows = np.array([ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,
        5,  6,  6,  6,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10, 10, 10, 11,
       11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
       17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22,
       22, 23, 23, 23],dtype='int32') #Row indices
        
        T_cols = np.array([ 0,  1,  2,  0,  1,  2,  0,  1,  2,  3,  4,  5,  3,  4,  5,  3,  4,
        5,  6,  7,  8,  6,  7,  8,  6,  7,  8,  9, 10, 11,  9, 10, 11,  9,
       10, 11, 12, 13, 14, 12, 13, 14, 12, 13, 14, 15, 16, 17, 15, 16, 17,
       15, 16, 17, 18, 19, 20, 18, 19, 20, 18, 19, 20, 21, 22, 23, 21, 22,
       23, 21, 22, 23],dtype='int32') #Column indices

        # flatten and then extend dirCos
        ex_dirCos = jnp.tile(dirCos.reshape(-1),8)

        # assign values to T compactly 
        T = T.at[T_rows,T_cols].set(ex_dirCos)
        
        # Return the transformation matrix.
        return T
    
    @staticmethod
    def J(new_xys, r, s):
        '''
        Returns the Jacobian matrix for the element
        '''
        # Get the local coordiantes, for jit-debugging purpose only 
        # Get the local coordinates for the element
        x1, y1, x2, y2, x3, y3, x4, y4 = new_xys

        # Return the Jacobian matrix
        return 1/4*jnp.array([[x1*(s + 1) - x2*(s + 1) + x3*(s - 1) - x4*(s - 1), y1*(s + 1) - y2*(s + 1) + y3*(s - 1) - y4*(s - 1)],
                          [x1*(r + 1) - x2*(r - 1) + x3*(r - 1) - x4*(r + 1), y1*(r + 1) - y2*(r - 1) + y3*(r - 1) - y4*(r + 1)]])

    @staticmethod
    def B_kappa(new_xys, r, s, J):
        # Get the local coordiantes, for jit-debugging purpose only 

        # For bending
        # Differentiate the interpolation functions
        # Row 1 = interpolation functions differentiated with respect to x
        # Row 2 = interpolation functions differentiated with respect to y
        # Note that the inverse of the Jacobian converts from derivatives with
        # respect to r and s to derivatives with respect to x and y
        dH = jnp.linalg.solve(J(new_xys,r,s), 1/4*jnp.array([[1 + s, -1 - s, -1 + s,  1 - s],                 
                                                  [1 + r,  1 - r, -1 + r, -1 - r]]))
        
        # Row 1 = d(beta_x)/dx divided by the local displacement vector 'u'
        # Row 2 = d(beta_y)/dy divided by the local displacement vector 'u'
        # Row 3 = d(beta_x)/dy + d(beta_y)/dx divided by the local displacement vector 'u'
        # Note that beta_x is a function of -theta_y and beta_y is a function of +theta_x (Equations 5.99, p. 423)
        B_kappa = jnp.array([[0,    0,     -dH[0, 0], 0,    0,     -dH[0, 1], 0,    0,     -dH[0, 2], 0,    0,     -dH[0, 3]],
                         [0, dH[1, 0],     0,     0, dH[1, 1],     0,     0, dH[1, 2],     0,     0, dH[1, 3],     0    ],
                         [0, dH[0, 0], -dH[1, 0], 0, dH[0, 1], -dH[1, 1], 0, dH[0, 2], -dH[1, 2], 0, dH[0, 3], -dH[1, 3]]])
        

        return B_kappa

    @staticmethod
    def B_gamma_MITC4(new_xys, r, s, J):
        '''
        Returns the [B] matrix for shear.
        MITC stands for mixed interpolation tensoral components. MITC elements
        are used in many programs and are known to perform well for thick and
        thin plates, and for distorted plate geometries.
        '''

        # Get the local coordinates for the element
        x1, y1, x2, y2, x3, y3, x4, y4 = new_xys
        x_axis = jnp.array([1, 0, 0]).T

        # Reference 2, Equations 5.105
        Ax = x1 - x2 - x3 + x4
        Bx = x1 - x2 + x3 - x4
        Cx = x1 + x2 - x3 - x4
        Ay = y1 - y2 - y3 + y4
        By = y1 - y2 + y3 - y4
        Cy = y1 + y2 - y3 - y4

        # Find the angles between the axes of the natural coordinate system and
        # the local x-axis.
        r_axis = jnp.array([(x1 + x4)/2 - (x2 + x3)/2, (y1 + y4)/2 - (y2 + y3)/2, 0.]).T
        s_axis = jnp.array([(x1 + x2)/2 - (x3 + x4)/2, (y1 + y2)/2 - (y3 + y4)/2, 0.]).T

        r_axis = r_axis/jnp.linalg.norm(r_axis)
        s_axis = s_axis/jnp.linalg.norm(s_axis)

        #alpha = arccos(dot(r_axis, x_axis))
        #beta = arccos(dot(s_axis, x_axis))
        
        # Reference 2, Equations 5.103 and 5.104 (p. 426)
        det_J = jnp.linalg.det(J(new_xys,r,s))

        gr = ((Cx + r*Bx)**2 + (Cy + r*By)**2)**0.5/(8*det_J)
        gs = ((Ax + s*Bx)**2 + (Ay + s*By)**2)**0.5/(8*det_J)

        # d      =           [    w1           theta_x1             theta_y1             w2            theta_x2              theta_y2            w3             theta_x3             theta_y3         w4             theta_x4             theta_y4      ]
        gamma_rz = gr*jnp.array([[(1 + s)/2, -(y1 - y2)*(1 + s)/4, (x1 - x2)*(1 + s)/4, -(1 + s)/2,  -(y1 - y2)*(1 + s)/4, (x1 - x2)*(1 + s)/4, -(1 - s)/2, -(y4 - y3)*(1 - s)/4, (x4 - x3)*(1 - s)/4, (1 - s)/2,  -(y4 - y3)*(1 - s)/4, (x4 - x3)*(1 - s)/4]])
        gamma_sz = gs*jnp.array([[(1 + r)/2, -(y1 - y4)*(1 + r)/4, (x1 - x4)*(1 + r)/4,  (1 - r)/2,  -(y2 - y3)*(1 - r)/4, (x2 - x3)*(1 - r)/4, -(1 - r)/2, -(y2 - y3)*(1 - r)/4, (x2 - x3)*(1 - r)/4, -(1 + r)/2, -(y1 - y4)*(1 + r)/4, (x1 - x4)*(1 + r)/4]])
        
        # Reference 2, Equations 5.102
        '''
        B_gamma_MITC4 = zeros((2, 12))
        B_gamma_MITC4[0, :] = gamma_rz*sin(beta) - gamma_sz*sin(alpha)
        B_gamma_MITC4[1, :] = -gamma_rz*cos(beta) + gamma_sz*cos(alpha)
        '''
        cos_alpha = jnp.dot(r_axis, x_axis) 
        cos_beta = jnp.dot(s_axis, x_axis)  
            
        # Non-differentiable when alpha/beta == 0  
        # which means we have to avoid the scenario when r-axis/s-axis is parallel to local x-axis
        # in conventional local coordinate, it is very common.
        # In our cooridinate system, the local 3-1 vector is the local x-axis. which can avoid the
        # aforementioned issue.
        # However, here the r-axis is clockwise with respect to the x-axis.
        # so change the sign of sin_alpha. TODO:??Really,yes......

        #Minus sign sin_alpha = -jnp.linalg.norm(jnp.cross(r_axis, x_axis)) 
        sin_alpha = -jnp.linalg.norm(jnp.cross(r_axis, x_axis)) 
        sin_beta = jnp.linalg.norm(jnp.cross(s_axis, x_axis)) 

        B_gamma_MITC4 = jnp.vstack((gamma_rz*sin_beta - gamma_sz*sin_alpha,-gamma_rz*cos_beta + gamma_sz*cos_alpha))
        
        # Return the [B] matrix for shear
        return B_gamma_MITC4
    
    @staticmethod
    def B_m(new_xys, r, s, J):
        # For membrane forces
        # Differentiate the interpolation functions
        # Row 1 = interpolation functions differentiated with respect to x
        # Row 2 = interpolation functions differentiated with respect to y
        # Note that the inverse of the Jacobian converts from derivatives with
        # respect to r and s to derivatives with respect to x and y
        dH = jnp.linalg.solve(J(new_xys,r,s), 1/4*jnp.array([[s + 1, -s - 1, s - 1, -s + 1],                 
                                                  [r + 1, -r + 1, r - 1, -r - 1]]))

        # Reference 2, Example 5.5 (page 353)
        B_m = jnp.array([[dH[0, 0],    0,     dH[0, 1],    0,     dH[0, 2],    0,     dH[0, 3],    0    ],
                     [   0,     dH[1, 0],    0,     dH[1, 1],    0,     dH[1, 2],    0,     dH[1, 3]],
                     [dH[1, 0], dH[0, 0], dH[1, 1], dH[0, 1], dH[1, 2], dH[0, 2], dH[1, 3], dH[0, 3]]])
        return B_m

    @staticmethod
    def Cb(nu,E,h):
        '''
        Returns the stress-strain matrix for plate bending.
        nu: poisson's ratio
        E: Young's modulus
        h: thickness of the shell
        '''

        Cb = (E*h**3/(12*(1 - nu**2)))*jnp.array([[1,  nu,      0    ],
                                            [nu, 1,       0    ],
                                            [0,  0,  (1 - nu)/2]])
        
        return Cb

    @staticmethod
    def Cs(nu,E,h):
        '''
        Returns the stress-strain matrix for shear.
        nu: poisson's ratio
        E: Young's modulus
        h: thickness of the shell
        '''
        # Reference 1, Equations (5.97), page 422
        k = 5/6

        Cs = (E*h*k/(2*(1 + nu)))*jnp.array([[1, 0],
                                       [0, 1]])

        return Cs

    @staticmethod
    def Cm(nu,E,kx_mod,ky_mod):
        """
        Returns the stress-strain matrix for an isotropic or orthotropic plane stress element
        """
        
        # Apply the stiffness modification factors for each direction to obtain orthotropic
        # behavior. Stiffness modification factors of 1.0 in each direction (the default) will
        # model isotropic behavior. Orthotropic behavior is limited to the element's local
        # coordinate system.
        Ex = E*kx_mod
        Ey = E*ky_mod
        nu_xy = nu
        nu_yx = nu

        # The shear modulus will be unafected by orthotropic behavior
        # Logan, Appendix C.3, page 750
        G = E/(2*(1 + nu))

        # Gallagher, Equation 9.3, page 251
        Cm = 1/(1 - nu_xy*nu_yx)*jnp.array([[   Ex,    nu_yx*Ex,           0         ],
                                        [nu_xy*Ey,    Ey,              0         ],
                                        [    0,        0,     (1 - nu_xy*nu_yx)*G]])
        
        return Cm

    @staticmethod
    def index_k_b():
        '''
        In JAX, for-loops are very slow.
        It is better to create an array for the indices without tracing in JAX and then assign the values to the array compactly 
        with array programming. 
        We go through this for loop using standard numpy -> static operations, non-traced

        ------
        This is for the local stiffness matrix for bending stresses (k_b)
        '''
        # Step through each term in the unexpanded stiffness matrix
        # i = Unexpanded matrix row
        
        i_vec = np.linspace(0,11,12,dtype='int32')
        j_vec = np.linspace(0,11,12,dtype='int32')
        i_arr,j_arr = np.meshgrid(i_vec,j_vec,indexing='ij')

        m_arr = np.array([[ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
       [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
       [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
       [ 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
       [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9],
       [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
       [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
       [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
       [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
       [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
       [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21],
       [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]])
        
        n_arr = np.array([[ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22],
       [ 2,  3,  4,  8,  9, 10, 14, 15, 16, 20, 21, 22]])
        
        return m_arr,n_arr,i_arr,j_arr

    @staticmethod
    def k_b(crds, t, E, nu, Cb, Cs, J, B_kappa, loc_crds, B_gamma_MITC4, index_k_b, kx_mod=1.0, ky_mod=1.0):
        '''
        Returns the local stiffness matrix for bending stresses
        '''
        # Get the local coordiantes 
        new_xys,normal,normal_tri1,normal_tri2 = loc_crds(crds) 

        Cb_val = Cb(nu,E,t)
        Cs_val = Cs(nu,E,t)

        # Define the gauss point for numerical integration
        gp = 1/3**0.5

        # Get the determinant of the Jacobian matrix for each gauss pointing 
        # Doing this now will save us from doing it twice below
        J1 = jnp.linalg.det(J(new_xys,gp,gp))
        J2 = jnp.linalg.det(J(new_xys,-gp,gp))
        J3 = jnp.linalg.det(J(new_xys,-gp,-gp))
        J4 = jnp.linalg.det(J(new_xys,gp,-gp))


        # Get the bending B matrices for each gauss point   
        B1_kappa = B_kappa(new_xys,gp,gp,J)
        B2_kappa = B_kappa(new_xys,-gp,gp,J)
        B3_kappa = B_kappa(new_xys,-gp,-gp,J)
        B4_kappa = B_kappa(new_xys,gp,-gp,J)
        
        # Create the stiffness matrix with bending stiffness terms  
        # See Reference 1, Equation 5.94    
        k1 = (jnp.matmul(B1_kappa.T, jnp.matmul(Cb_val, B1_kappa))*J1 + 
             jnp.matmul(B2_kappa.T, jnp.matmul(Cb_val, B2_kappa))*J2 +  
             jnp.matmul(B3_kappa.T, jnp.matmul(Cb_val, B3_kappa))*J3 +  
             jnp.matmul(B4_kappa.T, jnp.matmul(Cb_val, B4_kappa))*J4)   
        
        # Get the MITC4 shear B matrices for each gauss point   
        B1_gamma = B_gamma_MITC4(new_xys,gp,gp,J)
        B2_gamma = B_gamma_MITC4(new_xys,-gp,gp,J)
        B3_gamma = B_gamma_MITC4(new_xys,-gp,-gp,J)
        B4_gamma = B_gamma_MITC4(new_xys,gp,-gp,J)
        
        # Alternatively the shear B matrix below could be used. However, this matrix is prone to    
        # shear locking and will overestimate the stiffness.    
        # B1 = self.B_gamma(gp, gp) 
        # B2 = self.B_gamma(-gp, gp)    
        # B3 = self.B_gamma(-gp, -gp)   
        # B4 = self.B_gamma(gp, -gp)    
        # Add shear stiffness terms to the stiffness matrix 
        k2 = (jnp.matmul(B1_gamma.T, jnp.matmul(Cs_val, B1_gamma))*J1 + 
              jnp.matmul(B2_gamma.T, jnp.matmul(Cs_val, B2_gamma))*J2 + 
              jnp.matmul(B3_gamma.T, jnp.matmul(Cs_val, B3_gamma))*J3 + 
              jnp.matmul(B4_gamma.T, jnp.matmul(Cs_val, B4_gamma))*J4)  
        
        k = k1 + k2 

        k_rz = jnp.min(jnp.abs(jnp.array([k[1,1],k[2,2],k[4,4],k[5,5],k[7,7],k[8,8],k[10,10],k[11,11]])))/1000
        
        # Initialize the expanded stiffness matrix to all zeros
        k_exp = jnp.zeros((24, 24))

        #Get the indices
        m_arr,n_arr,i_arr,j_arr = index_k_b() # get the arrays for indices

        #Update k_exp from k
        k_exp = k_exp.at[m_arr,n_arr].set(k[i_arr,j_arr])  

        # Add the drilling degree of freedom's weak spring
        k_rz_i = np.array([5,11,17,23],dtype='int32')
        k_rz_j = np.array([5,11,17,23],dtype='int32')
        k_rz_vec = k_rz*np.ones(4)
        k_exp = k_exp.at[k_rz_i, k_rz_j].set(k_rz_vec)

        return k_exp

    @staticmethod
    def index_k_m():
        '''
        In JAX, for-loops are very slow.
        It is better to create an array for the indices without tracing in JAX and then assign the values to the array compactly 
        with array programming. 
        We go through this for loop using standard numpy -> static operations, non-traced
        
        ------
        This is for the local stiffness matrix for bending stresses (k_m)
        '''
        # Step through each term in the unexpanded stiffness matrix
        # i = Unexpanded matrix row
        i_vec = np.linspace(0,7,8,dtype='int32')
        j_vec = np.linspace(0,7,8,dtype='int32')
        i_arr,j_arr = np.meshgrid(i_vec,j_vec,indexing='ij')

        m_arr = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
       [ 1,  1,  1,  1,  1,  1,  1,  1],
       [ 6,  6,  6,  6,  6,  6,  6,  6],
       [ 7,  7,  7,  7,  7,  7,  7,  7],
       [12, 12, 12, 12, 12, 12, 12, 12],
       [13, 13, 13, 13, 13, 13, 13, 13],
       [18, 18, 18, 18, 18, 18, 18, 18],
       [19, 19, 19, 19, 19, 19, 19, 19]])
        
        n_arr = np.array([[ 0,  1,  6,  7, 12, 13, 18, 19],
       [ 0,  1,  6,  7, 12, 13, 18, 19],
       [ 0,  1,  6,  7, 12, 13, 18, 19],
       [ 0,  1,  6,  7, 12, 13, 18, 19],
       [ 0,  1,  6,  7, 12, 13, 18, 19],
       [ 0,  1,  6,  7, 12, 13, 18, 19],
       [ 0,  1,  6,  7, 12, 13, 18, 19],
       [ 0,  1,  6,  7, 12, 13, 18, 19]])

        return m_arr,n_arr,i_arr,j_arr
    

    @staticmethod
    def k_m(crds, t, E, nu, loc_crds, Cm, B_m, J, index_k_m, kx_mod=1.0, ky_mod=1.0):
        '''
        Returns the local stiffness matrix for membrane (in-plane) stresses.
        Plane stress is assumed
        '''
        # Get the local coordiantes 
        new_xys,normal,normal_tri1,normal_tri2 = loc_crds(crds) 

        Cm_val = Cm(nu,E,kx_mod,ky_mod)

        # Define the gauss point for numerical integration
        gp = 1/3**0.5

        # Get the membrane B matrices for each gauss point
        # Doing this now will save us from doing it twice below
        B1 = B_m(new_xys, gp, gp, J)
        B2 = B_m(new_xys, -gp, gp, J)
        B3 = B_m(new_xys, -gp, -gp, J)
        B4 = B_m(new_xys, gp, -gp, J)

        # See reference 1 at the bottom of page 353, and reference 2 page 466
        J1 = jnp.linalg.det(J(new_xys,gp,gp))
        J2 = jnp.linalg.det(J(new_xys,-gp,gp))
        J3 = jnp.linalg.det(J(new_xys,-gp,-gp))
        J4 = jnp.linalg.det(J(new_xys,gp,-gp))
        k = t*(jnp.matmul(B1.T, jnp.matmul(Cm_val, B1))*J1 +
               jnp.matmul(B2.T, jnp.matmul(Cm_val, B2))*J2 +
               jnp.matmul(B3.T, jnp.matmul(Cm_val, B3))*J3 +
               jnp.matmul(B4.T, jnp.matmul(Cm_val, B4))*J4)
        
        k_exp = jnp.zeros((24, 24))

        m_arr,n_arr,i_arr,j_arr = index_k_m()  # get the indices for assignment
        k_exp = k_exp.at[m_arr,n_arr].set(k[i_arr,j_arr]) 
        
        return k_exp