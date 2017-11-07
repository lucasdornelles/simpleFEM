# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:25:56 2017

@author: Lucas
"""

#2D FEA Benchmarking on python#

#INITIALIZING#

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


E=69*(10**9)
nu=0.3
h=0.01
gam=1.


"""
#isoparametric quadrilateral stiffness matrix#

psi=np.array([(1+nu)*gam, (1-(3*nu))*gam, 2+((1-(nu))*(gam**2)), (2*(gam**2))+(1-nu), ((1-nu)*(gam**2))-4, ((1-nu)*(gam**2))-1, (4*(gam**2))-(1-nu), (gam**2)-(1-nu)])

K01=((E*h)/(24*gam*(1-(nu**2))))*np.array([[ 4*psi[2], 3*psi[0], 2*psi[4],-3*psi[1],-2*psi[2],-3*psi[0],-4*psi[5], 3*psi[1]],
                                          [ 3*psi[0], 4*psi[3], 3*psi[1], 4*psi[7],-3*psi[0],-2*psi[3],-3*psi[1],-2*psi[6]],
                                          [ 2*psi[4], 3*psi[1], 4*psi[2],-3*psi[0],-4*psi[5],-3*psi[1],-2*psi[2], 3*psi[0]],
                                          [-3*psi[1], 4*psi[7],-3*psi[0], 4*psi[3], 3*psi[1],-2*psi[6], 3*psi[0],-2*psi[3]],
                                          [-2*psi[2],-3*psi[0],-4*psi[5], 3*psi[1], 4*psi[2], 3*psi[0], 2*psi[4],-3*psi[1]],
                                          [-3*psi[0],-2*psi[3],-3*psi[1],-2*psi[6], 3*psi[0], 4*psi[3], 3*psi[1], 4*psi[7]],
                                          [-4*psi[5],-3*psi[1],-2*psi[2], 3*psi[0], 2*psi[4], 3*psi[1], 4*psi[2],-3*psi[0]],
                                          [ 3*psi[1],-2*psi[6], 3*psi[0],-2*psi[3],-3*psi[1], 4*psi[7],-3*psi[0], 4*psi[3]]])
"""
#initialize iterations#

for power in range(1, 6):
    time.sleep(0.1)
    # GENERATING MESH#
    nelx=2**power
    nely=2**power
    
    dx=1
    dy=1
    
    print("numero de elementos: %d" %(nelx*nely))
    
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
        
    # Boundary conditions
    
    numpb=(2*(nely+1))+1
    iconh1=np.ndarray.tolist(np.zeros(numpb))
    iconh2=np.ndarray.tolist(np.zeros(numpb))
    icon=np.ndarray.tolist(np.zeros(numpb))
    rec=np.ndarray.tolist(np.zeros(numpb))
    
    # Definido as condicoes de contorno#
    # As condicoes de contorno sao colocadas no grau de liberdade#
    for i in range(numpb-1):
        iconh1[i]=i
    
    iconh1[numpb-1]=(nely+1)*(nelx+1)*2-1
    icon[numpb-1]=1
    rec[numpb-1]=-1*(10**8)
    
    freedofs=konedd
    for i in range(numpb):
        if icon[i]-1:
            freedofs.remove(iconh1[i])
    
    
    #Formar a matriz de rigidez global#
    
    print("Montando matriz global")
    start=time.time()
    
    KG=lil_matrix(((nely+1)*(nelx+1)*2,(nely+1)*(nelx+1)*2))
    
    for i in range(0,(nely*nelx)):
        
        e=i+1
        print("\r elemento analisado: " .format(e)+str(e), end="")
        i1=kone1[i]-1
        i2=kone2[i]-1
        i3=kone3[i]-1
        i4=kone4[i]-1
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
        
        KG[np.ix_((koned[i1]+koned[i2]+koned[i3]+koned[i4]),(koned[i1]+koned[i2]+koned[i3]+koned[i4]))]=KG[np.ix_((koned[i1]+koned[i2]+koned[i3]+koned[i4]),(koned[i1]+koned[i2]+koned[i3]+koned[i4]))]+K0
    
    end=time.time()
    print("")
    print("Tempo: %f" %(end-start))
    print("")
    
    #solucionar por selecao de index
    
    F=np.zeros(((nely+1)*(nelx+1)*2))
    KG=KG.tocsr()
    for i in range(numpb):
        if icon[i]:
            F[iconh1[i]]=rec[i]
    
    D=np.zeros(((nely+1)*(nelx+1)*2))
    
    print("Solucionando por selecao de index")
    
    start=time.time()
    D[freedofs]=spsolve(KG[np.ix_(freedofs,freedofs)],F[freedofs])
    end=time.time()
    print("")
    print("Tempo: %f" %(end-start))
    print("")
    
    #plotar malha deformada
    
    print("Plotando resultados")
    
    Dx=D[0:(nelx+1)*(nely+1)*2+1:2]
    Dy=D[1:(nelx+1)*(nely+1)*2+1:2]

    Dxx=Dx+np.array(x)
    Dyy=Dy+np.array(y)

    for i in range(0,(nelx*nely)):
        
        e=i+1
        print("\r elemento desenhado: " .format(e)+str(e), end="")
        
        plt.plot([Dxx[kone1[i]-1],Dxx[kone2[i]-1],Dxx[kone3[i]-1],Dxx[kone4[i]-1],Dxx[kone1[i]-1]],[Dyy[kone1[i]-1],Dyy[kone2[i]-1],Dyy[kone3[i]-1],Dyy[kone4[i]-1],Dyy[kone1[i]-1]], "-b")

    print("")
    print("end")
    print("")
    
    plt.show()
    
    
     

            
    
    
    
    
    
        
    