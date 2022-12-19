"""
References
---------
1. PyNite: An open-source FEA solver for python; https://github.com/JWock82/PyNite
2. Bathe, K. J. (2006). Finite element procedures. Klaus-Jurgen Bathe.
"""

#Import packages needed
from jax.numpy import array, zeros, matmul, sqrt
from jax.numpy.linalg import solve,norm
import jax.numpy as jnp
from jax import jacfwd,jit
from jax.tree_util import register_pytree_node_class
import numpy as np




@register_pytree_node_class  #Register 'BeamCol' objects as a valid jax type
class BeamCol():
    '''
    A class for an elastic beam-column element.
    We register this class to JAX, so that it is a valid JAX type class.

    Inputs
    -----
    eleTag: int
        The tag of this element

    i_nodeTag, j_nodeTag: int
        Node tag of i-node and j-node of the element

    x1,y1,z1,x2,y2,z2: float
        The coordinate of the i-th node and the j-th node of this element (xi,yi,zi,xj,yj,zj).

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
        
    def __init__(self, eleTag, i_nodeTag, j_nodeTag, x1, y1, z1, x2, y2, z2, E, G, Iy, Iz, J, A):
        
        #Inputs, Children of Pytree
        self.eleTag = eleTag    # Element tag
        self.i_nodeTag = i_nodeTag #i-node's tag
        self.j_nodeTag = j_nodeTag #j-node's tag
        self.x1,self.y1,self.z1,self.x2,self.y2,self.z2 = [x1,y1,z1,x2,y2,z2] #Coordinates
        self.E = E  # The modulus of elasticity of the element
        self.G = G  # The shear modulus of the element
        self.Iy = Iy  # The y-axis moment of inertia
        self.Iz = Iz  # The z-axis moment of inertia
        self.J = J  # The torsional constant
        self.A = A  # The cross-sectional area
                

    def L(self):
        '''
        Return the length of the beam-column
        '''
        return sqrt((self.x1 - self.x2)**2+(self.y1-self.y2)**2+(self.z1 - self.z2)**2)

    def T(self):
        '''
        Returns the transformation matrix between local and global axis.
        '''
        x1 = self.x1
        x2 = self.x2
        y1 = self.y1
        y2 = self.y2
        z1 = self.z1
        z2 = self.z2
        zero_entry = 0
        one_entry = 1
        C = array([(x2-x1)/self.L(), (y2-y1)/self.L(), (z2-z1)/self.L()]) #unit vector
    

        #Projections of C 
        Cxz_vec = array([(x2-x1)/self.L(), zero_entry, (z2-z1)/self.L()]) 
        Cxz = norm(Cxz_vec,axis=0)
        Cx = (x2-x1)/self.L() #projection on x
        Cy = (y2-y1)/self.L() #projection on y
        Cz = (z2-z1)/self.L() #projection on z
        sin_theta = Cxz
        cos_theta = Cy
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
        # !!!!Should be optimized in the future, less operations with .at.set
        transMatrix = zeros((12, 12))
        transMatrix = transMatrix.at[0:3,0:3].set(dirCos)
        transMatrix = transMatrix.at[3:6, 3:6].set(dirCos)
        transMatrix = transMatrix.at[6:9, 6:9].set(dirCos)
        transMatrix = transMatrix.at[9:12, 9:12].set(dirCos)
        return transMatrix


    def K_local(self):
        '''
        Return the element's stiffness matrix WITHOUT coordinate transformation
        '''
        E = self.E
        G = self.G
        Iy = self.Iy
        Iz = self.Iz
        J = self.J
        A = self.A
        L = self.L()
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
    
    def K(self):
        '''
        Return the element's stiffness matrix in global coordinate system

        '''
        return matmul(solve(self.T(),self.K_local()),self.T())
    

    #Tell JAX how to flatten this class
    def tree_flatten(self):
        children = (self.eleTag,self.i_nodeTag, self.j_nodeTag, self.x1,self.y1,self.z1,self.x2,self.y2,self.z2,self.E,self.G,self.Iy,self.Iz,self.J,self.A)
        aux_data = None
        return (children, aux_data)

    #Tell JAX how to unflatten this class
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    



'''
The following functions calculate the sensitivity of element's stiffness matrix K w.r.t its coordinates.

In order to implement Jax's 'Vmap' functionality later to boost the calculation of sensitivity, 
Note that functions are defined for one 'BeamCol' object, not a list of 'BeamCol' objects, this is because
'jax.vmap' works for objects(Python.containers), not containers of objects. 

These functions are:
    ----
    BeamCol_K(eleTag, x1, y1, z1, x2, y2, z2,E,G,Iy,Iz,J,A):
        Create a 'BeamCol' object and return the global stiffness matrix of this element
    ----
    
    ----
    Ele_Sens_K_Coord(BeamCol):
        Implement Jax.jacfwd to calculate the Jacobian.
        Return the sensitivity of the global stiffness of this member w.r.t. the coordinates of the member.
    ----

'''


#Stiffness matrix of the element
def BeamCol_K(eleTag, i_nodeTag, j_nodeTag, x1, y1, z1, x2, y2, z2,E,G,Iy,Iz,J,A):

    '''
    Return the element's local stiffness matrix (in global coordinates).

    
    Inputs:
        eleTag: int
            The tag of the element

        i_nodeTag: int
            The tag of i-node

        j_nodeTag: int
            The tag of j-node

        x1,y1,z1,x2,y2,z2: float
            The nodal coordiantes of the element

        E,G,Iy,Iz,J,A: float
            The properties of the element.

    Return:
        this_beamcol.K():   ndarray of shape (12,12)
            The stiffness matrix of the element.

    '''

    #Create a beam-column
    this_beamcol = BeamCol(eleTag, i_nodeTag, j_nodeTag, x1, y1, z1, x2, y2, z2, E, G, Iy, Iz, J, A)

    #Get the global stiffness matrix of this element
    return this_beamcol.K()

@jit
def Ele_K_to_Global(BeamCol):
    '''
    Return the [row, col, values] of element's local stiffness matrix in the global stiffness matrix.
    This will later be used for constructing the global stiffness matrix, which will be stored in a sparse matrix.
    Inputs:
        BeamCol:BeamCol() object
    
    Return:
        data: ndarray of shape(144,)
            The values of local stiffness matrix
        
        indices: ndarray of shape (144,2)
            The corresponding indices in global stiffness matrix.
            The first column is the row number, the second coclumn is the column number
    '''
    data = jnp.ravel(BeamCol.K()) #local stiffness matrix, flatten
    i =BeamCol.i_nodeTag #i-node
    j =BeamCol.j_nodeTag #j-node
    indices_dof = jnp.hstack((jnp.linspace(i*6,i*6+5,6,dtype=int),jnp.linspace(j*6,j*6+5,6,dtype=int))) #indices represented the dofs of this beamcol
    rows,cols = jnp.meshgrid(indices_dof,indices_dof,indexing='ij')#rows and columns
    indices = jnp.vstack((rows.ravel(),cols.ravel())).T #indices in global stiffness matrix
    return data,indices
    
@jit
def Ele_Sens_K_Coord(BeamCol):
    '''
    Return the sensitivity of element's 
    local stiffness matrix (in global coordinates) 
    w.r.t. the element's coordinates.

    
    Inputs:
        BeamCol: BeamCol() object

    Return:
        Sens_K_Coord: ndarray of shape (12,12,6)
            The derivatives of the stiffness matrix wrt to the nodal coordinates

    '''

    
    #Properties of this beam column
    eleTag = BeamCol.eleTag
    i_nodeTag, j_nodeTag = [BeamCol.i_nodeTag,BeamCol.j_nodeTag]
    x1,y1,z1,x2,y2,z2 = [BeamCol.x1,BeamCol.y1,BeamCol.z1,BeamCol.x2,BeamCol.y2,BeamCol.z2]
    E = BeamCol.E
    G = BeamCol.G
    Iy = BeamCol.Iy 
    Iz = BeamCol.Iz 
    J = BeamCol.J
    A = BeamCol.A
    
    #Calculate the sensitivity
    #argnums indicates the variables to which the Jacobian is calculated
    return jacfwd(BeamCol_K,argnums=(3,4,5,6,7,8))(eleTag, i_nodeTag, j_nodeTag, x1, y1, z1, x2, y2, z2, E, G, Iy, Iz, J, A)
