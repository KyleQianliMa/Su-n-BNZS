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

a=0.58
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

def p(u1,th1):
    return np.cos(u1)*np.matrix([1,0,0,0])+np.sin(u1)*np.matrix([0,0,0,1])+np.cos(th1)*np.matrix([0,1,0,0])+np.sin(th1)*np.matrix([0,0,1,0])


def exp(x,th1,S,th2,y):
    return p(x,th1) @ S @ p(y,th2).H

def expdbp(u1,th1,S1,J,S2,th2,u2):
    return exp(u1,th1,S1,th1,u1).H @ J @exp(u2,th2,S2,th2,u2)

g=5.0
h=0
g_perp=1.0

HKL=[0,0,2]

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

Spin={i+1:[] for i in range(len(S))}
angles={i+1:[] for i in range(len(S))}
###################################################

    
E=1e5
for h in np.arange(0.0,4.4,0.2):
    for i in range(0,10):
        # h=2
        s=0.0*np.pi
        t=2.0
        bnds = ((s, t*np.pi), (s, t*np.pi), (s, t*np.pi), (s, t*np.pi),(s, t*np.pi), (s, t*np.pi), (s, t*np.pi), (s, t*np.pi),
                (s, t*np.pi), (s, t*np.pi), (s, t*np.pi), (s, t*np.pi),(s, t*np.pi), (s, t*np.pi), (s, t*np.pi), (s, t*np.pi))
        initial=(t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), 
                 t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(),
                 t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), 
                 t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random(), t*np.pi*rdm.random())
        res = minimize(Energy,initial,bounds=bnds,tol=1e-5)
        print(h.round(1),i)
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
            
        else:
            continue

    energy.append(E)
    field.append(h)
    # Spin[1].append(Sp1[2].item().real)
    # Spin[2].append(Sp2[2].item().real)
    # Spin[3].append(Sp3[2].item().real)
    # Spin[4].append(Sp4[2].item().real)
    # Spin[5].append(Sp5[2].item().real)
    # Spin[6].append(Sp6[2].item().real)
    # Spin[7].append(Sp7[2].item().real)
    # Spin[8].append(Sp8[2].item().real)
    # Spin[9].append(Sp9[2].item().real)
    # Spin[10].append(Sp10[2].item().real)
    # Spin[11].append(Sp11[2].item().real)
    # Spin[12].append(Sp12[2].item().real)
    # Spin[13].append(Sp13[2].item().real)
    # Spin[14].append(Sp14[2].item().real)
    # Spin[15].append(Sp15[2].item().real)
    # Spin[16].append(Sp16[2].item().real)
    
    Spin[1].append(np.array(Sp1[2].real))
    Spin[2].append(np.array(Sp2[2].real))
    Spin[3].append(np.array(Sp3[2].real))
    Spin[4].append(np.array(Sp4[2].real))
    Spin[5].append(np.array(Sp5[2].real))
    Spin[6].append(np.array(Sp6[2].real))
    Spin[7].append(np.array(Sp7[2].real))
    Spin[8].append(np.array(Sp8[2].real))
    Spin[9].append(np.array(Sp9[2].real))
    Spin[10].append(np.array(Sp10[2].real))
    Spin[11].append(np.array(Sp11[2].real))
    Spin[12].append(np.array(Sp12[2].real))
    Spin[13].append(np.array(Sp13[2].real))
    Spin[14].append(np.array(Sp14[2].real))
    Spin[15].append(np.array(Sp15[2].real))
    Spin[16].append(np.array(Sp16[2].real))
    
    for i in range(16):
        angles[i+1].append(new_res.x[i])
        
    bulk_m.append((Sp1+Sp2+Sp3+Sp4+Sp5+Sp6+Sp7+Sp8+Sp9+Sp10+Sp11+Sp12+Sp13+Sp14+Sp15+Sp16)[2].item().real)
    # bulk_m.append( ((Sp4[2]+Sp5[2]+Sp11[2]+Sp14[2]+Sp3[2]+Sp6[2]+Sp12[2]+Sp13[2])**2 + (Sp1[2]+Sp8[2]+Sp10[2]+Sp15[2]+Sp2[2]+Sp7[2]+Sp9[2]+Sp16[2])**2).item().real)
    stagger_m.append(( Sp4+Sp5+Sp11+Sp14-Sp3-Sp6-Sp12-Sp13)[2].item().real)
    # bulk_m.append(( Sp4[2]+Sp5[2]+Sp11[2]+Sp14[2]+Sp3[2]+Sp6[2]+Sp12[2]+Sp13[2]).item().real)

#%%################################################
#             Flipping signs
##################################################

Tpt=10 #The Tpt-th data point where transition happens
for i in range(Tpt):
    for j in (3,6,12,13):
        Spin[j][i]=-np.abs(Spin[j][i])
    
    for j in (4,5,11,14):
        Spin[j][i]=np.abs(Spin[j][i])

# for i in range(22):
#     for j in (1,15,2,16):
#         Spin[j][i]=np.abs(Spin[j][i])

#     for j in (7,9,8,10):
#         Spin[j][i]=np.abs(Spin[j][i])


#%%################################################
#             Plotting results
#################################################
low=-0.5
upp=0.5
fig,ax=plt.subplots(4,3,figsize=(12,14),constrained_layout=True)
fig.suptitle(f'a={a},b={b},c={c},d={d}')
ax[0,0].plot(field,Spin[3],field,Spin[6],'.')
ax[0,0].set_title("spin3,6")
ax[0,0].set_ylim(low,upp)

ax[0,1].plot(field,Spin[12],field,Spin[13],'.')
ax[0,1].set_title("spin12,13")
ax[0,1].set_ylim(low,upp)

ax[0,2].plot(field,Spin[11],field,Spin[14],'.')
ax[0,2].set_title("spin11,14")
ax[0,2].set_ylim(low,upp)

ax[1,0].plot(field,Spin[4],field,Spin[5],'.')
ax[1,0].set_title("spin4,5")
ax[1,0].set_ylim(low,upp)

ax[1,1].plot(field,Spin[1],field,Spin[15],'.')
ax[1,1].set_title("spin1,15")
ax[1,1].set_ylim(low,upp)

ax[1,2].plot(field,Spin[2],field,Spin[16],'.')
ax[1,2].set_title("spin2,16")
ax[1,2].set_ylim(low,upp)

ax[2,0].plot(field,Spin[7],field,Spin[9],'.')
ax[2,0].set_title("spin7,9")
ax[2,0].set_ylim(low,upp)

ax[2,1].plot(field,Spin[8],field,Spin[10],'.')
ax[2,1].set_title("spin8,10")
ax[2,1].set_ylim(low,upp)

ax[2,2].plot(field,np.abs(bulk_m))
ax[2,2].set_title("Bulk magnetization")

ax[3,0].plot(field,stagger_m)
ax[3,0].set_title("Stagger magnetization")

ax[3,1].plot(field,energy,'.-')
ax[3,1].set_title("System Energy")

bm=np.abs(bulk_m)**2
bm=bm[-1]
sm=np.array(stagger_m)**2
sm=sm[0]
k=bm/sm

ax[3,2].plot(field, (np.abs(bulk_m))**2,color='red')
ax[3,2].plot(field, k*(np.array(stagger_m))**2,color='blue')
ax[3,2].set_title("Overlay Stagger and Bulk Magnetization squared")


fig,ax=plt.subplots(4,4,figsize=(12,14),constrained_layout=True)
fig.suptitle(f'a={a},b={b},c={c},d={d}')
ct=1
tit=0
keylist=list(Hprime.keys())
for i in range(4):
    for j in range(4):
        ax[i,j].plot(field,180*np.array(angles[ct])/np.pi,'.-')
        ax[i,j].set_ylim(0,360)
        if tit<8:
            ax[i,j].set_title(keylist[tit])
        else:ax[i,j].set_title(keylist[tit-8])
        tit+=1
        ct+=1