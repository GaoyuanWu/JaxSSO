# -*- coding: utf-8 -*-
"""
The code creates SAP2000 model for Kinetic Umbrella based on the parameters given to get the dynamic structural response.
It reads the pressure time history in pressure.csv as the input.
@author: Gaoyuan Wu
"""
#%%
# Preparatons: Importing Basic packages needed
import numpy as np  # Numerical packages
import matplotlib.pyplot as plt  # plotting
import pandas as pd # Pandas, transitions between csv/xlsx to python
import os
import sys
from scipy.signal import savgol_filter
from scipy.optimize import fsolve
gvty_water = 9.7706 # KN per m^3, density times GA g 

# Function that transform local para to Cartesian coordinates for given lambda and beta
def lbdToCart(lbd,beta,gamma,theta_star,b1,b2,c1,c2,r): #Note that lbd is an array
    Cart_coord = np.array(np.ones(3)) #Initialization of Cartesian 
    k1 = np.sqrt((b1**2) + (r*(1-lbd[0]))**2)
    k2 = np.sqrt((b2**2) + (r*(1-lbd[0]))**2) #k1 & k2
    theta_prime = np.arcsin(r*(1-lbd[0])/k1) + np.arcsin(r*(1-lbd[0])/k2) # Angle of upper panel
    theta_1 = theta_star - np.arctan(r * (1 - lbd[0]) / b1)
    theta_2 = theta_1 + theta_prime
    #print(lbd)
    #print(beta)
    #print(gamma)
    if gamma == 1: #Left panel
        c = c1
    if gamma == 2: #Right panel
        c = -c2
    if beta == 1:
        Cart_coord[0] = k1 * lbd[1] * np.cos(theta_1) #X
        Cart_coord[1] = c * lbd[0] #Y
        Cart_coord[2] = k1 * lbd[1] * np.sin(theta_1) #Z
    if beta == 2:
        Cart_coord[0] = k2 * lbd[1] * np.cos(theta_2) + k1 * np.cos(theta_1) #X
        Cart_coord[1] = c * lbd[0] #Y
        Cart_coord[2] = k2 * lbd[1] * np.sin(theta_2) + k1 * np.sin(theta_1)#Z
    return Cart_coord,k1,k2,theta_1,theta_2
        
def Macaulay(A): #Note that lbd is an array
    if A >= 0:
        return A
    else:
        return 0
    
def getCoordNForce(n_c1,n_c2,n_b1,n_b2,dw,theta_star,r,c1,c2,b1,b2,Hw,hb,d,Lw):
    gvty_water = 9.7706 # KN per m^3, density times GA g 
    # Before implementing lbdToCart function ,we have to get our lbd, beta & gamma first based on our SAP2000 settings
    # Nodes indexing, we specify bottom left node as NODE 1, and the indexing increases to the right; when reaching the right edge, it starts from the left at the adjacent row above

    nodes_num = (n_c1 + n_c2 + 1) * (n_b1 + n_b2 + 1) #Number of nodes
    nodes_coord = np.array(np.zeros((nodes_num,3))) #Array storing all nodes' coordinates
    lbd = np.array(np.zeros((nodes_num,2))) #Array storing all nodes' lambda (local para)
    gamma = np.array(np.zeros((nodes_num))) #Array storing all nodes' gamma (local para)
    beta = np.array(np.zeros((nodes_num))) #Array storing all nodes' beta (local para)
    k1 = np.array(np.zeros((nodes_num))) #Array storing all nodes' k1
    k2 = np.array(np.zeros((nodes_num))) #Array storing all nodes' k2
    theta_1 = np.array(np.zeros((nodes_num))) #Array storing all nodes' theta_1
    theta_2 = np.array(np.zeros((nodes_num))) #Array storing all nodes' theta_2
    
    x1 = np.array(np.zeros((nodes_num,3))) #Array storing tributary nodes
    x2 = np.array(np.zeros((nodes_num,3))) #Array storing tributary nodes
    x3 = np.array(np.zeros((nodes_num,3))) #Array storing tributary nodes
    x4 = np.array(np.zeros((nodes_num,3))) #Array storing tributary nodes
    
    #Local vector,initilization
    uk = np.array(np.zeros((nodes_num,3)))
    vk = np.array(np.zeros((nodes_num,3)))
    
    #Unit vector:
    nk = np.array(np.zeros((nodes_num,3)))
    
    #Pressure initialization:
    StaticP_val = np.array(np.zeros((nodes_num)))        #Static component, value
    StaticP_vec = np.array(np.zeros((nodes_num,3)))      #Static Component, vector
    StaticP_Area = np.array(np.zeros((nodes_num,3)))     #Static Component times tributary area
    

    WaveP_val = np.array(np.zeros((nodes_num)))        #Wave component, value
    WaveP_vec = np.array(np.zeros((nodes_num,3)))      #Wave Component, vector
    WaveP_Area = np.array(np.zeros((nodes_num,3)))     #Wave Component times tributary area
    
    Tri_A = np.array(np.zeros((nodes_num))) #Tributary Area
    
    # Get local para for nodes and transform to Cartesian
    for i in range (nodes_num):
        if 0 <= i % (n_c1 + n_c2 + 1) and i % (n_c1 + n_c2 + 1) <= n_c1:
            gamma[i] = 1
            lbd[i,0] = 1 - (i % (n_c1 + n_c2 + 1))/n_c1 #lambda 1
        if i % (n_c1 + n_c2 + 1) > n_c1:
            gamma[i] = 2
            lbd[i,0] = ((i % (n_c1 + n_c2 + 1)) - n_c1)/n_c2 #lambda 2
        if i <= ((n_c1 + n_c2 + 1) * (n_b1+1) - 1):
            beta[i] = 1
            lbd[i,1] = (int(i / (n_c1 + n_c2 + 1)))/ (n_b1)  #which row
        if i > ((n_c1 + n_c2 + 1) * (n_b1+1) - 1):
            beta[i] = 2 
            lbd[i,1] = ((int(i / (n_c1 + n_c2 + 1))) - n_b1) / n_b2
        i_coord,k1[i],k2[i],theta_1[i],theta_2[i] = lbdToCart(lbd[i,:],beta[i],gamma[i],theta_star,b1,b2,c1,c2,r)
        nodes_coord[i,0] = round(i_coord[0] , 3) 
        nodes_coord[i,1] = round(i_coord[1] , 3)
        nodes_coord[i,2] = round(i_coord[2] , 3)# Storing Cartesian coordinates 
        
    ####################### Coordinates Finished#############################
    
    ####################### Let's comupte pressure###########################
    
    
    ####################### Hydrostatic Component#########################
        lbd_10 = np.array([[0],[lbd[i,1]]]) #Set lbd1 = 0
        lbd_11 = np.array([[1],[lbd[i,1]]]) #Set lbd1 = 1
        lbd_21 = np.array([[lbd[i,0]],[1]]) #Set lbd2 = 1
        lbd_20 = np.array([[lbd[i,0]],[0]]) #Set lbd2 = 0
        vk[i,:] = (lbdToCart(lbd_21,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0] - (lbdToCart(lbd_20,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0] #Local vector vk
        
        if gamma[i] == 1: #Left panel
            uk[i,:] = (lbdToCart(lbd_10,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0] - (lbdToCart(lbd_11,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0] #Local vector uk
        if gamma[i] == 2: #Right Panel
            uk[i,:] = (lbdToCart(lbd_11,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0] - (lbdToCart(lbd_10,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0] #Local vector uk
   
        cross_p = np.cross(vk[i,:],uk[i,:])
        nk[i,:] = cross_p / np.linalg.norm(cross_p) # Unit vector  
        #Compute hydrostatic pressure on node
        h = np.sqrt(b1**2+r**2) * np.sin(theta_star - np.arctan(r/b1)) #Verte height
        if beta[i] == 1:
            if (dw - k1[i] * lbd[i,1] * np.sin(theta_1[i])) >= 0:
                StaticP_val[i] = gvty_water * (dw - k1[i] * lbd[i,1] * np.sin(theta_1[i]))
            else:
                StaticP_val[i] = 0
        if beta[i] == 2:
            if (dw - (r * lbd[i,0] * np.cos(theta_star) + h + k2[i] * lbd[i,1] * np.sin(theta_2[i]))) >= 0:
                StaticP_val[i] = gvty_water * (dw - (r * lbd[i,0] * np.cos(theta_star) + h + k2[i] * lbd[i,1] * np.sin(theta_2[i])))
            else:
                StaticP_val[i] = 0
                
        StaticP_vec[i,:] = StaticP_val[i] * nk[i,:] # Pressure vector at nodes
        
        # Compute acting area of the pressure
        if gamma[i] == 1:
            delta_lbd1 = 1 / (2*n_c1) #Left Panel diff
        if gamma[i] == 2:
            delta_lbd1 = 1 / (2*n_c2) #Right Panel diff
        if beta[i] == 1:
            delta_lbd2 = 1 / (2*n_b1) #Lower Panel diff
        if beta[i] == 2:
            delta_lbd2 = 1 / (2*n_b2) #Upper Panel diff
        
        if lbd[i,0] == 1 : #Lbd1 = 1 
            if lbd[i,1] == 1:
                tp_lbd = lbd[i,:]
                x1[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0]],[lbd[i,1] - delta_lbd2]])
                x4[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            if lbd[i,1] == 0:
                tp_lbd = np.array([[lbd[i,0]],[lbd[i,1] + delta_lbd2]])
                x1[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = lbd[i,:]
                x4[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            else:
                tp_lbd = np.array([[lbd[i,0]],[lbd[i,1] + delta_lbd2]])
                x1[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0]],[lbd[i,1] - delta_lbd2]])
                x4[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            
        if lbd[i,0]!= 1:
            if lbd[i,1] == 1:
                tp_lbd = np.array([[lbd[i,0] + delta_lbd1],[lbd[i,1]]])
                x1[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0] + delta_lbd1],[lbd[i,1] - delta_lbd2]])
                x4[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            if lbd[i,1] == 0:
                tp_lbd = np.array([[lbd[i,0] + delta_lbd1],[lbd[i,1] + delta_lbd2]])
                x1[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0] + delta_lbd1],[lbd[i,1]]])
                x4[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            else:
                tp_lbd = np.array([[lbd[i,0] + delta_lbd1],[lbd[i,1] + delta_lbd2]])
                x1[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0] + delta_lbd1],[lbd[i,1] - delta_lbd2]])
                x4[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
        
        if lbd[i,0] == 0:
            if lbd[i,1] == 1:
                tp_lbd = lbd[i,:]
                x2[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0]],[lbd[i,1] - delta_lbd2]])
                x3[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            if lbd[i,1] == 0:
                tp_lbd = np.array([[lbd[i,0]],[lbd[i,1] + delta_lbd2]])
                x2[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = lbd[i,:]
                x3[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            else:
                tp_lbd = np.array([[lbd[i,0]],[lbd[i,1] + delta_lbd2]])
                x2[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0]],[lbd[i,1] - delta_lbd2]])
                x3[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
        if lbd[i,0] != 0:
            if lbd[i,1] == 1:
                tp_lbd = np.array([[lbd[i,0] - delta_lbd1],[lbd[i,1]]])
                x2[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0] - delta_lbd1],[lbd[i,1] - delta_lbd2]])
                x3[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            if lbd[i,1] == 0:
                tp_lbd = np.array([[lbd[i,0] - delta_lbd1],[lbd[i,1] + delta_lbd2]])
                x2[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0] - delta_lbd1],[lbd[i,1]]])
                x3[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
            else:
                tp_lbd = np.array([[lbd[i,0] - delta_lbd1],[lbd[i,1] + delta_lbd2]])
                x2[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
                tp_lbd = np.array([[lbd[i,0] - delta_lbd1],[lbd[i,1] - delta_lbd2]])
                x3[i] = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[0]
        
        A_k = 0.5 * np.linalg.norm(np.cross((x2[i] - x4[i]),(x1[i] - x3[i]))) # Area of the quad
        Tri_A[i] = A_k
        StaticP_Area[i,:] = A_k * StaticP_vec[i,:]
        
        ########################## Wave Component########################################
        #print("node",i+1)
        #print("gamma=",gamma[i])
        #print("beta=",beta[i])
        #print("lbd=",lbd[i,:])
        #print("k1=",k1[i])
        #print("k2=",k2[i])
        #print("theta_1=",theta_1[i]*180/np.pi)
        #print("theta_2=",theta_2[i]*180/np.pi)
        #print("z=" ,nodes_coord[i,2])
        #print("dw=",dw)
        #print("dw+yita_star",(dw + 1.5 * Hw))
        # The Goda pressure coefficients
        alpha_1 = 0.6 + 0.5 * ((4 * np.pi * d)/(np.sinh(4 * np.pi * d / Lw) * Lw)) ** 2
        alpha_2 = min(((hb - dw)/(3 * hb)) * (Hw / dw)**2, 2 * dw / Hw)
        alpha_3 = 1 - dw/d * (1 - 1/np.cosh(2 * np.pi * d/Lw))

        # The Goda wave pressure distribution
        p1 = (alpha_1 + alpha_2) * gvty_water * Hw
        p3 = alpha_3 * p1
        #print("alpha3 =",alpha_3)
        # Parameters used based on Geometry 
        tp_lbd = np.array([[0], [lbd[i,1]]]) #Set lbd1 == 0
        tp_k1,none,tp_theta1 = (lbdToCart(tp_lbd,beta[i],gamma[i],theta_star,b1,b2,c1,c2,r))[1:4]
        h = tp_k1 * np.sin(tp_theta1) 
        hv = r * lbd[i,0] * np.cos(theta_star) + h
        #print("h=",h)
        #print("hv=",hv)

        # Parameters used based on wave para
        yita_star = 1.5 * Hw
        l1_prime = Macaulay(min(hv,dw + yita_star) - dw)
        l1_prime = l1_prime / np.sin(theta_1[i])
        l2_prime = Macaulay(dw + yita_star - max(hv,dw))
        l2_prime = l2_prime / np.sin(theta_2[i])
        l1 = min(dw,hv) / np.sin(theta_1[i])
        l2 = Macaulay(dw - hv) / np.sin(theta_2[i])
        #print("l1=",l1)
        #print("l2=",l2)
        #print("l1_prime=",l1_prime)
        #print("l2_prime=",l2_prime,"\n")

        ######## Wave Component Values #######
        if beta[i] == 1: 
            tp_wave1 = p3 + (p1 - p3) / (l1 + l2) * lbd[i,1] * k1[i]
            tp_wave2 = (p1 * (l1  + l2 + l1_prime + l2_prime) - p1 * lbd[i,1] * k1[i]) / (l1_prime + l2_prime)
            tp_waveratio = ((l1  + l2 + l1_prime + l2_prime) - lbd[i,1] * k1[i]) / (l1_prime + l2_prime)
            tp_wave = min(tp_wave1,tp_wave2)
            #print("wave1" ,tp_wave1)
            #print("wave2" ,tp_wave2)
            #print("2ratio",tp_waveratio)
            #print("tp_wave",tp_wave)
            WaveP_val[i] = Macaulay(tp_wave)
            #print(WaveP_val[i],'\n')
        if beta[i] == 2:
            tp_wave1 = p3 + (p1 - p3) / (l1 + l2) * (lbd[i,1] * k2[i] + k1[i])
            tp_wave2 = ((p1 * (l1  + l2 + l1_prime + l2_prime) - p1 * (lbd[i,1] * k2[i] + k1[i])) / (l1_prime + l2_prime))
            tp_waveratio = ((l1  + l2 + l1_prime + l2_prime) - (lbd[i,1] * k2[i] + k1[i])) / (l1_prime + l2_prime)
            tp_wave = min(tp_wave1,tp_wave2)
            #print("wave1" ,tp_wave1)
            #print("wave2" ,tp_wave2)
            #print("2ratio",tp_waveratio)
            #print("tp_wave",tp_wave)
            WaveP_val[i] = Macaulay(tp_wave)
            #print(WaveP_val[i],'\n')

            #WaveP_val[i] = Macaulay(min(p3 + (p1 - p3) / (l1 + l2) * (lbd[i,1] * k2[i] + k1[i]), (p1 * (l1  + l2 + l1_prime + l2_prime) - p1 * (lbd[i,1] * k2[i] + k1[i]) / (l1_prime + l2_prime))))
        
        WaveP_vec[i,:] = WaveP_val[i] * nk[i,:] # Wave Pressure vector at nodes
        WaveP_Area[i,:] = A_k * WaveP_vec[i,:]  # Wave Pressure times tributary area


    return nodes_coord, StaticP_Area, StaticP_val, StaticP_vec, WaveP_Area, WaveP_vec, WaveP_val,nk,Tri_A


def getShellConnectivity(n_c1,n_c2,n_b1,n_b2):
    area_num = (n_c1 + n_c2) * (n_b1 + n_b2) #Area numbers
    area_cnct = np.array(np.zeros((area_num,4))) #Array storing connectivity of Shell elements
    
    for i in range(area_num):
        row = int (i/(n_c1 + n_c2)) + 1 #Which row the area is in
        area_cnct[i,0] = int((row-1)*(n_c1 + n_c2 + 1) + 1 + (i - (row -1)*(n_c1 + n_c2))) #Joint 1 of the shell element
        area_cnct[i,1] = int(area_cnct[i,0] + (n_c1 + n_c2 + 1))#Joint 2 of the shell element
        area_cnct[i,2] = int(area_cnct[i,0] + (n_c1 + n_c2 + 2))#Joint 3 of the shell element
        area_cnct[i,3] = int(area_cnct[i,0] + 1) #Joint 4 of the shell element
        
    return area_cnct
