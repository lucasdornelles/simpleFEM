# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:25:56 2017

@author: Lucas
"""

#2D FEA on python#

#INITIALIZING#

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from iqlsm import localmatrix
from qmeshgenerator import meshgenerator

E=69*(10**9)
nu=0.3
h=0.01
gam=1.

dx, dy, nelx, nely, numnp, x, y, kone1, kone2, kone3, kone4, koned, konedd=meshgenerator()

# Boundary conditions

numpb=(2*(nely+1))+1
iconh1=np.ndarray.tolist(np.zeros(numpb))
iconh2=np.ndarray.tolist(np.zeros(numpb))
icon=np.ndarray.tolist(np.zeros(numpb))
rec=np.ndarray.tolist(np.zeros(numpb))

# Defining boundary conditions#
# The boundary condition are defined on nodes degrees of freedom#

for i in range(numpb-1):
    iconh1[i]=i

iconh1[numpb-1]=(nely+1)*(nelx+1)*2-1
icon[numpb-1]=1
rec[numpb-1]=-1*(10**6)

freedofs=konedd
for i in range(numpb):
    if icon[i]-1:
        freedofs.remove(iconh1[i])
    
    
#Setting global stiffness matrix#
    
print("Setting global matrix")
start=time.time()

KG=lil_matrix(((nely+1)*(nelx+1)*2,(nely+1)*(nelx+1)*2))

for i in range(0,(nely*nelx)):
    
    e=i+1
    print("\r analised element: " .format(e)+str(e), end="")
    i1=kone1[i]-1
    i2=kone2[i]-1
    i3=kone3[i]-1
    i4=kone4[i]-1
   
    K0=localmatrix (x, y, nu, E, h, i1, i2, i3, i4)
    
    KG[np.ix_((koned[i1]+koned[i2]+koned[i3]+koned[i4]),(koned[i1]+koned[i2]+koned[i3]+koned[i4]))]=KG[np.ix_((koned[i1]+koned[i2]+koned[i3]+koned[i4]),(koned[i1]+koned[i2]+koned[i3]+koned[i4]))]+K0

end=time.time()
print("")
print("Time: %f" %(end-start))
print("Done")
print("")

#Simple solver

F=np.zeros(((nely+1)*(nelx+1)*2))
KG=KG.tocsr()
for i in range(numpb):
    if icon[i]:
        F[iconh1[i]]=rec[i]

D=np.zeros(((nely+1)*(nelx+1)*2))

print("Using a simple solver")

start=time.time()
D[freedofs]=spsolve(KG[np.ix_(freedofs,freedofs)],F[freedofs])
end=time.time()
print("")
print("Time: %f" %(end-start))
print("Done")
print("")

#ploting deslocated mesh

print("Ploting results")

Dx=D[0:(nelx+1)*(nely+1)*2+1:2]
Dy=D[1:(nelx+1)*(nely+1)*2+1:2]

Dxx=Dx+np.array(x)
Dyy=Dy+np.array(y)

for i in range(0,(nelx*nely)):
    
    e=i+1
    print("\r drawing element: " .format(e)+str(e), end="")
    
    plt.plot([Dxx[kone1[i]-1],Dxx[kone2[i]-1],Dxx[kone3[i]-1],Dxx[kone4[i]-1],Dxx[kone1[i]-1]],[Dyy[kone1[i]-1],Dyy[kone2[i]-1],Dyy[kone3[i]-1],Dyy[kone4[i]-1],Dyy[kone1[i]-1]], "-b")

print("")
print("done")
print("")

plt.axes().set_aspect('equal')
plt.show()
   
    
        
    
