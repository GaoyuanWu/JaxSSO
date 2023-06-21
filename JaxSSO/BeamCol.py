"""
References
---------
1. PyNite: An open-source FEA solver for python; https://github.com/JWock82/PyNite
2. Bathe, K. J. (2006). Finite element procedures. Klaus-Jurgen Bathe.
3. http://www.gcg.ufjf.br/pub/doc132.pdf (Transformation Matrix)
"""
#%%
#Import packages needed
import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial


#%%
@register_pytree_node_class  #Register 'BeamCol' objects as a valid jax type
class BeamCol():
    '''
    A class for an elastic beam-column element.
    We register this class to JAX, so that it is a valid JAX type class.

    Inputs
    -----
    eleTag: int
        The tag of this element

    ele_crds: jnp.array of shape (6,)
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
        '''
        x1,y1,z1,x2,y2,z2 = crds
        length = jnp.sqrt((x1 - x2)**2+(y1-y2)**2+(z1 - z2)**2) #the length
        zero_entry = 0
        one_entry = 1
        C = jnp.array([(x2-x1)/length, (y2-y1)/length, (z2-z1)/length]) #unit vector
    

        #Projections of C 
        Cxz_vec = jnp.array([(x2-x1)/length, zero_entry, (z2-z1)/length]) 
        Cxz = jnp.linalg.norm(Cxz_vec,axis=0)
        Cx = (x2-x1)/length #projection on x
        Cy = (y2-y1)/length #projection on y
        Cz = (z2-z1)/length #projection on z
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
    def K(T,K_local):
        '''
        Return the element's stiffness matrix in global coordinate system.
        Shape of (12,12).

        '''

        return jnp.matmul(jnp.linalg.solve(T,K_local),T)
    
    def K_call(self):
        xyzs = self.ele_crds

        T_K = BeamCol.T(xyzs) #Coordinate transformation
        K_loc = BeamCol.K_local(xyzs,self.E,self.G,self.Iy,self.Iz,self.J,self.A) #Local stiffness matrix
        K_eleG = BeamCol.K(T_K,K_loc) #Element's stiffness matrix in global coordinates, yet to be assigned

        return K_eleG

    #Tell JAX how to flatten this class
    def tree_flatten(self):
        children = (self.eleTag, self.i_nodeTag, self.j_nodeTag, self.ele_crds, self.E, self.G, self.Iy, self.Iz, self.J, self.A)
        aux_data = None
        return (children, aux_data)

    #Tell JAX how to unflatten this class
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    



