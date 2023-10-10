import numpy as np
import random as rdm
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import pyswarms as ps
from scipy.optimize import minimize
import pdb
a=b=7.83940
c=13.61310
alpha=beta=gamma=np.pi/2
V=a*b*c*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
a1=[a,0,0]
a2=[0,b,0]
a3=[0,0,c]

b1=2*np.pi*np.cross(a2,a3)/V
b2=2*np.pi*np.cross(a3,a1)/V
b3=2*np.pi*np.cross(a1,a2)/V
pi=np.pi

muB=0.05788

def FF(Q):
    A=0.0540
    a=25.0293
    B=0.3101
    b=12.1020
    C=0.6575
    c=4.7223
    dd=-0.0216
    s=Q/(4*np.pi)
    return A*np.exp(-a*s**2)+B*np.exp(-b*s**2)+C*np.exp(-c*s**2)+dd

sx=np.matrix([[0,1/2],
              [1/2,0]])
sy=np.matrix([[0,-1j/2],
              [1j/2,0]])
sz=np.matrix([[1/2,0],
              [0,-1/2]])

idd=np.matrix([[1,0],
              [0,1]])
#---------------------------------
S1=np.array([np.kron(sx,idd),
             np.kron(sy,idd),
             np.kron(sz,idd)])
S2=np.array([np.kron(idd,sx),
              np.kron(idd,sy),
              np.kron(idd,sz)])
#---------------------------------
S3=np.array([np.kron(sx,idd),
             np.kron(sy,idd),
             np.kron(sz,idd)])
S4=np.array([np.kron(idd,sx),
              np.kron(idd,sy),
              np.kron(idd,sz)])
#---------------------------------
S5=np.array([np.kron(sx,idd),
             np.kron(sy,idd),
             np.kron(sz,idd)])
S6=np.array([np.kron(idd,sx),
              np.kron(idd,sy),
              np.kron(idd,sz)])
#---------------------------------
S7=np.array([np.kron(sx,idd),
             np.kron(sy,idd),
             np.kron(sz,idd)])
S8=np.array([np.kron(idd,sx),
              np.kron(idd,sy),
              np.kron(idd,sz)])
#---------------------------------


###############################
#   Eigenfunction
###############################

def phi(mu1):
    return np.cos(mu1)*np.matrix([1,0,0,0])+np.sin(mu1)*np.matrix([0,0,0,1])

###############################
#   Hamiltonian Matrix
###############################
def Jprime(A,B,C,D,E,F,Sa,Sb):
    
    Jmat=np.array([[A,B,C],
                   [B,D,E],
                   [C,E,F]])
    ham=0
    for i in range(0,3):
        for j in range(0,3):
            ham=ham+Jmat[i,j]*np.matmul(Sa[i],Sb[j])
    ev, ef=np.linalg.eig(ham) #eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
    ef=ef[:,np.argsort(ev)].real
    ev=np.sort(ev)
    return ev,ef,ham

# def Jdbprime(A,B,C,D,E,F,Sa,Sb):
#     Jmat=np.array([[A,B,C],
#                    [B,D,E],
#                    [C,E,F]])
#     ham=0
#     for i in range(0,3):
#         for j in range(0,3):
#             ham=ham+Jmat[i,j]*np.matmul(Sa[i],Sb[j])
#     ev, ef=np.linalg.eig(ham) #eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
#     ef=ef[:,np.argsort(ev)].real
#     ev=np.sort(ev)
#     return ev,ef,ham
Jdbprime=np.array([[0    ,   0 , 0   ],
                   [0    ,   0 , 0   ],
                   [0    ,   0 , 0.58]])


_,_,Jp12=Jprime(-0.52,0,0,0.52,0,-0.72,S1,S2)
_,_,Jp34=Jprime(-0.52,0,0,0.52,0,-0.72,S3,S4)
_,_,Jp56=Jprime(-0.52,0,0,0.52,0,-0.72,S5,S6)
_,_,Jp78=Jprime(-0.52,0,0,0.52,0,-0.72,S7,S8)
##########Creating empy measured space#############
energy=[]
field=[]
stagger_m=[]
bulk_m=[]

Spin1=[]
Spin2=[]
Spin3=[]
Spin4=[]
Spin5=[]
Spin6=[]
Spin7=[]
Spin8=[]
###################################################

def FEnergy(params):
    mu1=params[0]
    mu2=params[1]
    mu3=params[2]
    mu4=params[3]
    FE_perp=phi(mu1) @ Jp12 @ phi(mu1).H + g *muB* h * (phi(mu1) @ (S1[2] + S2[2]) @ phi(mu1).H) +\
            (phi(mu1) @ S2 @ phi(mu1).H).H @ Jdbprime  @ (phi(mu2) @ S3 @ phi(mu2).H) + (phi(mu1) @ S1 @ phi(mu1).H).H @ Jdbprime  @ (phi(mu2) @ S4 @ phi(mu2).H) + \
            phi(mu2) @ Jp34 @phi(mu2).H + g*muB*h*(phi(mu2) @ S3[2] @phi(mu2).H+ phi(mu2)@ S4[2] @phi(mu2).H)
    
    FE_para=phi(mu3) @ Jp56 @ phi(mu3).H + gp*muB * h * (phi(mu3) @ (S5[2] + S6[2]) @ phi(mu3).H) +\
            (phi(mu4) @ S8 @ phi(mu4).H).H @ Jdbprime  @ (phi(mu3) @ S5 @ phi(mu3).H) + (phi(mu4) @ S7 @ phi(mu4).H).H @ Jdbprime  @ (phi(mu3) @ S6 @ phi(mu3).H) + \
            phi(mu4) @ Jp78 @phi(mu4).H + gp*muB*h*(phi(mu4) @ S7[2] @phi(mu4).H+ phi(mu2)@ S8[2] @phi(mu2).H)
        
    FE=(FE_perp+FE_para).item().real
    return FE

E=1e5
h=90
for i in range(0,50):
    # h=2
    g=5.12
    gp=1.2         
    bnds = ((0.0, 2.0*np.pi), (0.0, 2.0*np.pi),(0.0, 2.0*np.pi),(0.0, 2.0*np.pi))
    initial=(2*np.pi*rdm.random(),2*np.pi*rdm.random(),2*np.pi*rdm.random(),2*np.pi*rdm.random())
    res = minimize(FEnergy,initial,bounds=bnds,tol=1e-5)
    print(h,i)
    if res.fun<=E:
        new_res=res
        E=res.fun
        Sp1=phi(new_res.x[0])@S1@phi(new_res.x[0]).H
        Sp2=phi(new_res.x[0])@S2@phi(new_res.x[0]).H
        
        Sp3=phi(new_res.x[1])@S3@phi(new_res.x[1]).H
        Sp4=phi(new_res.x[1])@S4@phi(new_res.x[1]).H
        
        Sp5=phi(new_res.x[2])@S5@phi(new_res.x[2]).H
        Sp6=phi(new_res.x[2])@S6@phi(new_res.x[2]).H
        
        Sp7=phi(new_res.x[3])@S7@phi(new_res.x[3]).H
        Sp8=phi(new_res.x[3])@S8@phi(new_res.x[3]).H
        
    else:
        continue
    
energy.append(E)
field.append(h)
bulk_m.append((Sp1[2]+Sp2[2]+Sp3[2]+Sp4[2]+Sp5[2]+Sp6[2]+Sp7[2]+Sp8[2]).item().real)
stagger_m.append(((Sp1[2]+Sp2[2]-Sp3[2]-Sp4[2])).item().real)
Spin1.append(Sp1[2].item().real)
Spin2.append(Sp2[2].item().real)
Spin3.append(Sp3[2].item().real)
Spin4.append(Sp4[2].item().real)
Spin5.append(Sp5[2].item().real)
Spin6.append(Sp6[2].item().real)
Spin7.append(Sp7[2].item().real)
Spin8.append(Sp8[2].item().real)
    
#################################################
#             Plotting results
#################################################
# fig,ax=plt.subplots(3,2,figsize=(10,11))
# ax[0,0].plot(field,Spin1,field,Spin2,'.')
# ax[0,0].set_title("spin1,2")

# ax[0,1].plot(field,Spin3,field,Spin4,'.')
# ax[0,1].set_title("spin3,4")

# ax[1,0].plot(field,Spin5,field,Spin6,'.')
# ax[1,0].set_title("spin5,6")

# ax[1,1].plot(field,Spin7,field,Spin8,'.')
# ax[1,1].set_title("spin7,8")

# ax[2,0].plot(field,bulk_m)
# ax[2,0].set_title("Bulk magnetization")

# ax[2,1].plot(field,stagger_m)
# ax[2,1].set_title("Stagger magnetization")
# plt.show()

# plt.plot(field,energy,'.-')
# plt.show()
print("Sp1=",Spin1)
print("Sp2=",Spin2)
print("Sp3=",Spin3)
print("Sp4=",Spin4)

# print("Sp5=",Spin5)
# print("Sp6=",Spin6)
# print("Sp7=",Spin7)
# print("Sp8=",Spin8)

