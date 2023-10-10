import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
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

S1=np.array([np.kron(sx,idd),
             np.kron(sy,idd),
             np.kron(sz,idd)])

S2=np.array([np.kron(idd,sx),
              np.kron(idd,sy),
              np.kron(idd,sz)])


def TT(A,C,B):
    
    Jmat=np.array([[A,C,0],[C,A,0],[0,0,B]])
    ham=0
    for i in range(0,3):
        for j in range(0,3):
            ham=ham+Jmat[i,j]*np.matmul(S1[i],S2[j])
    ev, ef=np.linalg.eig(ham)
    ef=ef[:,np.argsort(ev)]
    ev=np.sort(ev)
    return ev,ef,ham

def kron_delta(a,b):
    if a==b:
        return 1
    else:
        return 0



def SQ(h,k,l,v,atom1,atom2,A,C,B):
    T=TT(A,C,B)
    ef=T[1]
    S=np.array([S1,S2])
    R=np.array([atom1,atom2])
    Omega=0
    Q=np.array([h,k,l]) @ np.array([b1,b2,b3])
    Qnorm=np.linalg.norm(Q)
    
    
    for a in range(0,3):
        for b in range(0,3):
            for m in range(0,2):
                for n in range(0,2):
                    Omega=Omega+((kron_delta(a,b)-Q[a]*Q[b]/(Qnorm+0.001)**2) *  np.exp(1j*2*pi*np.dot( [h,k,l],(R[m]-R[n])) ) *\
                                np.conj(ef[:,0]).T @ S[m][a] @ ef[:,v] * np.conj(ef[:,v]).T @S[n][b] @ ef[:,0] )
    Omega=Omega*FF(Qnorm)**2
    return Omega
R2=[0.8417,0.3417,0]
R3=[1.1583,0.6583,0]

R1=[0.3417,0.1583,0]
R4=[0.6583,-0.1583,0]
A=-0.667
C=0.223
B=0.624

def hkl_SQ(atom1,atom2,atom3,atom4,A,C,B,v):
    h=np.linspace(-4,4,100)
    k=np.linspace(-4,4,100)
    I_mat=np.zeros([len(h),len(k)])
    for i in range(len(h)):
        for j in range(len(k)):
            I_mat[i,j]=SQ(h[i],k[j],0,v,atom1,atom2, A, C, B)+SQ(h[i],k[j],0,v,atom3,atom4, A, -C, B)
    
    plt.imshow(I_mat,origin='lower',cmap='rainbow')
    # plt.savefig(r'C:\Users\qmc\OneDrive\ONRL\Data\level3.png', format='png',dpi=400)

# hkl_SQ(R2,R3,R1,R4,A,C,B,1)

def powder(v):
    Q_powder=[]
    I_powder=[]
    h=np.linspace(-2,2,11)
    k=np.linspace(-2,2,11)
    l=np.linspace(-4,4,11)
    counter=0
    for x in range(len(h)):
        for y in range(len(k)):
            for z in range(len(l)):            
                Q=np.array([h[x],k[y],l[z]]) @ np.array([b1,b2,b3])
                I_powder.append((SQ(h[x],k[y],l[z],v,R2,R3, A, C, B)+SQ(h[x],k[y],l[z],v,R1,R4, A, -C, B)).real)
                Q_powder.append(np.linalg.norm(Q))
                counter=counter+1
                print(counter)
    
    Q_powder=np.array(Q_powder)
    I_powder=np.array(I_powder)
    Q_s=np.sort(Q_powder)
    I_s=I_powder[np.argsort(Q_powder)]
    
    Int,Q,_=binned_statistic(Q_s, I_s,statistic='mean', bins=30)
    Qplot=[]
    for i in range(len(Q)-1):
        Qplot.append( (Q[i]+Q[i+1])/2 )
    plt.plot(Qplot,Int,'.')
    save=np.transpose([Qplot,Int])
    np.savetxt("QI_level %i.txt" %v,save)
    return Q, Int

# pwd2=powder(1)
