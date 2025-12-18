import numpy as np
import JaxSSO.model as Model

def fea_problem_model():
    '''
    Define a testing FEA problem with a benchmark result from SAP2000, a commercial FEA software widely used in industry.

    This problem solves a sinply-supported 2D framed-arch under gravity loads.

    '''
    # %%
    #Nodes
    n_node = 100
    Q = 500 #Nodal load
    rise =  5 #Rise
    x_span = 10
    x_nodes = np.linspace(0,x_span,n_node)
    y_nodes = np.zeros(n_node)
    z_nodes = -(rise/(x_span**2/4))*((x_nodes-x_span/2)**2 - x_span**2/4)#parabolic arch
    z_nodes[0] = 0
    z_nodes[n_node-1] = 0
    design_nodes = np.array([i for i in range(n_node) if i!=0 and i!=n_node-1])
    #Connectivity
    n_ele = n_node -1 #number of elements
    cnct = np.zeros((n_ele,2),dtype=int) #connectivity matrix
    x_ele = np.zeros((n_ele,2))
    y_ele = np.zeros((n_ele,2))
    z_ele = np.zeros((n_ele,2))
    for i in range(n_ele):
        cnct[i,0] = i
        cnct[i,1] = i+1
        x_ele[i,:] = [x_nodes[i],x_nodes[i+1]]
        y_ele[i,:] = [y_nodes[i],y_nodes[i+1]]
        z_ele[i,:] = [z_nodes[i],z_nodes[i+1]]

    #Sectional properties-> 600x400 rectangle

    E = 1.999E+08#Young's modulus (Gpa)
    G = E/(2*(1+0.3)) #Shear modolus-> E = 2G(1+mu)
    Iy = 6.572e-05 #Moement of inertia in m^4
    Iz = 3.301e-06 #Same, about z axis
    J = Iy + Iz	#Polar moment of inertia
    A = 4.265e-03 #Area

    #%%
    #Create model
    model = Model.Model() #model for sensitivity analysis

    #Adding nodes and boundary conditions
    for i in range(n_node):
        model.add_node(i,x_nodes[i],y_nodes[i],z_nodes[i])
        if i not in design_nodes:
            model.add_support(i,[1,1,1,1,0,1]) #Pinned, only Ry allow
        else:
            model.add_nodal_load(i,nodal_load=[0.0,0.0,-Q,0.0,0.0,0.0])

    #Adding elements
    for i in range(n_ele):
        i_node = cnct[i,0]
        j_node = cnct[i,1]
        model.add_beamcol(i,i_node,j_node,E,G,Iy,Iz,J,A) 

    return model

    



