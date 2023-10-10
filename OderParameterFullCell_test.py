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

sx=np.matrix([[0,1/2],
              [1/2,0]])
sy=np.matrix([[0,-1j/2],
              [1j/2,0]])
sz=np.matrix([[1/2,0],
              [0,-1/2]])

idd=np.matrix([[1,0],
              [0,1]])
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

def p(u1,th1):
    return np.exp(1j*th1)*np.cos(u1)*np.matrix([1,0,0,0])+np.sin(u1)*np.matrix([0,0,0,1])

# def p(u1):
    # return np.cos(u1)*np.matrix([1,0,0,0])+np.sin(u1)*np.matrix([0,0,0,1])

def Jprime(Jmat,Sa,Sb):
    
    ham=0
    for i in range(0,3):
        for j in range(0,3):
            ham=ham+Jmat[i,j]*np.matmul(Sa[i],Sb[j])
    ev, ef=np.linalg.eig(ham) #eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
    ef=ef[:,np.argsort(ev)].real
    ev=np.sort(ev)
    return ev,ef,ham

S={}

num_of_spins=32
for i in (1,2,3,4,7,8,11,12):
    S1=np.array([np.kron(sx,idd),
                 np.kron(sy,idd),
                 np.kron(sz,idd)])
    S2=np.array([np.kron(idd,sx),
                  np.kron(idd,sy),
                  np.kron(idd,sz)])
    S={**S,i:S1}
    # print(i)

for i in (15,16,6,5,9,10,14,13):
    S2=np.array([np.kron(idd,sx),
                  np.kron(idd,sy),
                  np.kron(idd,sz)])
    S={**S,i:S2}
    # print(i)

R={1:[0.17085,   0.07915,   1.00000],
   2:[0.67085,   0.07915,   1.00000],
   3:[0.42085,   0.17085,   1.00000],
   4:[0.92085,   0.17085,   1.00000],
   5:[0.07915,   0.32915,   1.00000],
   6:[0.57915,   0.32915,   1.00000],
   7:[0.32915,   0.42085,   1.00000],
   8:[0.82915,   0.42085,   1.00000],
   9:[0.17085,   0.57915,   1.00000],
  10:[0.67085,   0.57915,   1.00000],
  11:[0.42085,   0.67085,   1.00000],
  12:[0.92085,   0.67085,   1.00000],
  13:[0.07915,   0.82915,   1.00000],
  14:[0.57915,   0.82915,   1.00000],
  15:[0.32915,   0.92085,   1.00000]}

a=0.52
b=0.72
c=0.55
d=0.0

Jdbp =  np.array([ [0    ,   0 ,     0],
                   [0    ,   0 ,     0],
                   [0    ,   0 ,     c]])

Jmat =   np.array([[-a   ,   0 ,     0],
                   [0    ,   a ,     0],
                   [0    ,   0 ,    -b]])

Jmatrotx=np.array([[a    ,   0 ,     0],
                   [0    ,  -a ,     0],
                   [0    ,   0 ,    -b]])

Jdm =    np.array([[0    ,   0 ,     0],
                   [0    ,   0 ,     0],
                   [0    ,   0 ,    -d]])

Hprime = {"B3-6":Jprime(Jmat,S[3],S[6])[2],
          "B4-5":Jprime(Jmat,S[4],S[5])[2],
          "B11-14":Jprime(Jmat,S[11],S[14])[2],
          "B12-13":Jprime(Jmat,S[12],S[13])[2],#parallel to the field
       
          "B1-15":Jprime(Jmatrotx,S[1],S[15])[2], 
          "B2-16":Jprime(Jmatrotx,S[2],S[16])[2],
          "B7-9":Jprime(Jmatrotx,S[7],S[9])[2],
          "B8-10":Jprime(Jmatrotx,S[8],S[10])[2],#perpendicular to the field
       }

def exp(x,th1,S,th2,y):
    return p(x,th1) @ S @ p(y,th2).H

def expdbp(u1,th1,S1,J,S2,th2,u2):
    return exp(u1,th1,S1,th1,u1).H @ J @exp(u2,th2,S2,th2,u2)

g=5.0
# h=0
g_perp=1.0

HKL=[0,0,2]

def moment(M,R1,h,k,l):
    Q=np.array([h,k,l]) @ np.array([b1,b2,b3])
    return M*np.exp(1j*2*pi*np.dot(Q, R1))

def Energy(params):
    u1=params[0]
    u2=params[1]
    u3=params[2]
    u4=params[3]
    u5=params[4]
    u6=params[5]
    u7=params[6]
    u8=params[7]
    
    th1=params[8]
    th2=params[9]
    th3=params[10]
    th4=params[11]
    th5=params[12]
    th6=params[13]
    th7=params[14]
    th8=params[15]    

    Eintra = 0
    for i, keys in zip(range(len(Hprime)), Hprime):
                   Eintra = Eintra + exp(params[i],params[i+8],Hprime[keys],params[i+8],params[i])
                   
    Einter = expdbp(u5,th5,S[1],Jdbp,S[7],th7,u7) + expdbp(u6,th6,S[2],Jdbp,S[8],th8,u8) +\
             expdbp(u1,th1,S[3],Jdbp,S[5],th2,u2) + expdbp(u2,th2,S[4],Jdbp,S[6],th1,u1) +\
             expdbp(u7,th7,S[9],Jdbp,S[15],th5,u5) + expdbp(u8,th8,S[10],Jdbp,S[16],th6,u6) +\
             expdbp(u3,th3,S[11],Jdbp,S[13],th4,u4) + expdbp(u4,th4,S[12],Jdbp,S[14],th3,u3) +\
             expdbp(u5,th5,S[1],Jdbp,S[16],th6,u6) + expdbp(u6,th6,S[2],Jdbp,S[15],th5,u5) +\
             expdbp(u1,th1,S[3],Jdbp,S[14],th3,u3) + expdbp(u2,th2,S[4],Jdbp,S[13],th4,u4) +\
             expdbp(u2,th2,S[5],Jdbp,S[12],th4,u4) + expdbp(u1,th1,S[6],Jdbp,S[11],th3,u3) +\
             expdbp(u7,th7,S[7],Jdbp,S[10],th8,u8) + expdbp(u8,th8,S[8],Jdbp,S[9],th7,u7)
                      
    Ezeeman_para = exp(u2,th2, S[4][2],th2, u2) + exp(u1,th1, S[3][2],th1, u1) + exp(u1,th1, S[6][2],th1, u1) + exp(u4,th4, S[12][2],th4, u4) +\
                   exp(u2,th2, S[5][2],th2, u2) + exp(u3,th3, S[11][2],th3, u3) + exp(u3,th3, S[14][2],th3, u3) + exp(u4,th4, S[13][2],th4, u4)
    
    Ezeeman_para=g*muB*h* Ezeeman_para
    
    Ezeeman_perp = exp(u6,th6, S[2][0],th6, u6) + exp(u8,th8, S[8][0],th8, u8) + exp(u5,th5, S[1][0],th5, u5) + exp(u8,th8, S[10][0],th8, u8) +\
                   exp(u7,th7, S[7][0],th7, u7) + exp(u6,th6, S[16][0],th6, u6) + exp(u7,th7, S[9][0],th7, u7) + exp(u5,th5, S[15][0],th5, u5)
    Ezeeman_perp=g_perp*muB*h* Ezeeman_perp
    
    # E_dm1= expdbp(u5,S[1],Jdm,S[4],u2) +  expdbp(u5,S[1],Jdm,S[13],u4) +\
    #       expdbp(u6,S[2],Jdm,S[14],u3) + expdbp(u6,S[2],Jdm,S[3],u1) +\
    #       expdbp(u1,S[3],Jdm,S[15],u5) + expdbp(u2,S[4],Jdm,S[16],u6) +\
    #       expdbp(u2,S[5],Jdm,S[8],u8) +  expdbp(u2,S[5],Jdm,S[9],u7) +\
    #       expdbp(u1,S[6],Jdm,S[7],u7) +  expdbp(u1,S[6],Jdm,S[10],u8) +\
    #       expdbp(u7,S[7],Jdm,S[11],u3) +  expdbp(u8,S[8],Jdm,S[12],u4) +\
    #       expdbp(u7,S[9],Jdm,S[12],u4) +  expdbp(u8,S[10],Jdm,S[11],u3) +\
    #       expdbp(u4,S[13],Jdm,S[16],u6) +  expdbp(u3,S[14],Jdm,S[15],u5) 
          
    # E_dm2= expdbp(u5,S[1],Jdm,S[3],u1) +  expdbp(u5,S[1],Jdm,S[5],u2) +\
    #        expdbp(u6,S[2],Jdm,S[4],u2) + expdbp(u6,S[2],Jdm,S[6],u1) +\
    #        expdbp(u1,S[3],Jdm,S[7],u7) + expdbp(u2,S[4],Jdm,S[8],u8) +\
    #        expdbp(u2,S[5],Jdm,S[7],u7) +  expdbp(u1,S[6],Jdm,S[8],u8) +\
    #        expdbp(u7,S[9],Jdm,S[11],u3) +  expdbp(u7,S[9],Jdm,S[13],u4) +\
    #        expdbp(u8,S[10],Jdm,S[12],u4) +  expdbp(u8,S[10],Jdm,S[14],u3) +\
    #        expdbp(u3,S[11],Jdm,S[15],u5) +  expdbp(u4,S[12],Jdm,S[16],u6) +\
    #        expdbp(u4,S[13],Jdm,S[15],u5) +  expdbp(u3,S[14],Jdm,S[16],u6) 
      
    return (Eintra + Einter + Ezeeman_para + Ezeeman_perp).item().real
##########Creating empy measured space#############

energy=[]
field=[]
stagger_m=[]
bulk_m=[]

# Spin1=[]
# Spin2=[]
# Spin3=[]
# Spin4=[]
# Spin5=[]
# Spin6=[]
# Spin7=[]
# Spin8=[]
# Spin9=[]
# Spin10=[]
# Spin11=[]
# Spin12=[]
# Spin13=[]
# Spin14=[]
# Spin15=[]
# Spin16=[]
Spin={i+1:[] for i in range(len(S))}
###################################################


    
E=1e5
h=2
s=0.0*np.pi
t=2.0
bnds = ((s, t*np.pi), (s, t*np.pi), (s, t*np.pi), (s, t*np.pi),(s, t*np.pi), (s, t*np.pi), (s, t*np.pi), (s, t*np.pi),
        (s, t*np.pi), (s, t*np.pi), (s, t*np.pi), (s, t*np.pi),(s, t*np.pi), (s, t*np.pi), (s, t*np.pi), (s, t*np.pi))
initial=(t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), 
         t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(),
         t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), 
         t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random())
res = minimize(Energy,initial,bounds=bnds,tol=1e-7)
if res.fun<=E:
    new_res=res
    E=res.fun
    Sp3=p(new_res.x[0],new_res.x[0+8])@S[3]@p(new_res.x[0],new_res.x[0+8]).H
    Sp6=p(new_res.x[0],new_res.x[0+8])@S[6]@p(new_res.x[0],new_res.x[0+8]).H
    
    Sp4=p(new_res.x[1],new_res.x[1+8])@S[4]@p(new_res.x[1],new_res.x[1+8]).H
    Sp5=p(new_res.x[1],new_res.x[1+8])@S[5]@p(new_res.x[1],new_res.x[1+8]).H
    
    Sp11=p(new_res.x[2],new_res.x[2+8])@S[11]@p(new_res.x[2],new_res.x[2+8]).H
    Sp14=p(new_res.x[2],new_res.x[2+8])@S[14]@p(new_res.x[2],new_res.x[2+8]).H
    
    Sp12=p(new_res.x[3],new_res.x[3+8])@S[12]@p(new_res.x[3],new_res.x[3+8]).H
    Sp13=p(new_res.x[3],new_res.x[3+8])@S[13]@p(new_res.x[3],new_res.x[3+8]).H
    
    Sp1=p(new_res.x[4],new_res.x[4+8])@S[1]@p(new_res.x[4],new_res.x[4+8]).H
    Sp15=p(new_res.x[4],new_res.x[4+8])@S[15]@p(new_res.x[4],new_res.x[4+8]).H
    
    Sp2=p(new_res.x[5],new_res.x[5+8])@S[2]@p(new_res.x[5],new_res.x[5+8]).H
    Sp16=p(new_res.x[5],new_res.x[5+8])@S[16]@p(new_res.x[5],new_res.x[5+8]).H
    
    Sp7=p(new_res.x[6],new_res.x[6+8])@S[7]@p(new_res.x[6],new_res.x[6+8]).H
    Sp9=p(new_res.x[6],new_res.x[6+8])@S[9]@p(new_res.x[6],new_res.x[6+8]).H
    
    Sp8=p(new_res.x[7],new_res.x[7+8])@S[8]@p(new_res.x[7],new_res.x[7+8]).H
    Sp10=p(new_res.x[7],new_res.x[7+8])@S[10]@p(new_res.x[7],new_res.x[7+8]).H


    energy.append(E)
    field.append(h)

print('Sp1=',Sp1[2].item().real)
print('Sp7=',Sp7[2].item().real)
print('Sp6=',Sp6[2].item().real)
print('Sp4=',Sp4[2].item().real)
        

