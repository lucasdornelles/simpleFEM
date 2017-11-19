# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:31:42 2017

@author: Lucas
"""
import numpy as np

def localmatrix (x, y, nu, E, h, i1, i2, i3, i4):
    
    xs=np.array([-0.5773,0.5773,0.5773,-0.5773])
    ys=np.array([-0.5773,-0.5773,0.5773,0.5773])

    K0=np.zeros((8,8))

    Ee=(E/(1-(nu**2)))*(np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]))

    for j in range(4):
        for l in range(4):
               
            Jac=np.array([[(1/4*((x[i2]-x[i1])*(1-ys[l])+(x[i3]-x[i4])*(1+ys[l]))), (1/4*((y[i2]-y[i1])*(1-ys[l])+(y[i3]-y[i4])*(1+ys[l])))],
                      [(1/4*((x[i4]-x[i1])*(1-xs[j])+(x[i3]-x[i2])*(1+xs[j]))), (1/4*((y[i4]-y[i1])*(1-xs[j])+(y[i3]-y[i2])*(1+xs[j])))]])
                
                    
            Dnd=np.array([[(-(1/4)*(1-ys[l])), ((1/4)*(1-ys[l])), ((1/4)*(1+ys[l])), (-(1/4)*(1+ys[l]))],
                          [(-(1/4)*(1-xs[j])), (-(1/4)*(1+xs[j])), ((1/4)*(1+xs[j])), ((1/4)*(1-xs[j]))]])
                
            Dnd[(0,1),(0,0)]=np.dot(np.linalg.inv(Jac), Dnd[(0,1),(0,0)])
            Dnd[(0,1),(1,1)]=np.dot(np.linalg.inv(Jac), Dnd[(0,1),(1,1)])
            Dnd[(0,1),(2,2)]=np.dot(np.linalg.inv(Jac), Dnd[(0,1),(2,2)])
            Dnd[(0,1),(3,3)]=np.dot(np.linalg.inv(Jac), Dnd[(0,1),(3,3)])

            B=np.array([[Dnd[0,0], 0, Dnd[0,1], 0, Dnd[0,2], 0, Dnd[0,3], 0],
                        [0, Dnd[1,0], 0, Dnd[1,1], 0, Dnd[1,2], 0, Dnd[1,3]],
                        [Dnd[1,0], Dnd[0,0], Dnd[1,1], Dnd[0,1], Dnd[1,2], Dnd[0,2], Dnd[1,3], Dnd[0,3]]])

            K0=K0+h*np.dot((np.dot(np.ndarray.transpose(B), Ee)), B)*np.linalg.det(Jac)

    return K0;