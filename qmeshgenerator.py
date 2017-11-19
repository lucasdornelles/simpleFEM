# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:33:03 2017

@author: Lucas
"""

import numpy as np

def meshgenerator ():
    print("Generating mesh with strutured mesh generator")
    dx=float(input("Lenght: "))
    dy=float(input("Height: "))
    nelx=int(input("Number of elements in x: "))
    nely=int(input("Number of elements in y: "))
        
    numnp=(nelx+1)*(nely+1)
        
    x=sorted((nely+1)*(np.ndarray.tolist(np.arange(0,(dx+(dx/nelx)),(dx/nelx)))))
    y=(nelx+1)*(np.ndarray.tolist(-np.arange(-dy,0+(dy/nely),dy/nely)))
        
    kone4=list(range(1,int((nely+1)*(nelx)+1)))
    kone1=list(range(1,int((nely+1)*(nelx)+1)))
    kone3=list(range(nely+2,(nely+1)*(nelx+1)+1))
    kone2=list(range(nely+2,(nely+1)*(nelx+1)+1))
    for i in range(1, int(nelx+1)):
        kone4.remove(int((nely+1)*i))
        kone1.remove(int((1+((nely+1)*(i-1)))))
        kone3.remove(int(nely+1+((nely+1)*i)))
        kone2.remove(int(nely+2+((nely+1)*(i-1))))
            
        konedd=list(range(0,int((2*(nely+1)*(nelx+1)))))
        koned=list()
    for i in range(0,int((2*(nely+1)*(nelx+1))),2):
        koned.append(konedd[i:i+2])
        
    return dx, dy, nelx, nely, numnp, x, y, kone1, kone2, kone3, kone4, koned, konedd;