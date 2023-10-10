from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
# import pdb
from SSL_BNZS_Field import SQ,TT,kron_delta
#-----------Initiate MPI---------------
world_comm = MPI.COMM_WORLD
proc_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()
#-----------Lattice Parameter----------
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

sx=np.matrix([[0,1/2],
              [1/2,0]])
sy=np.matrix([[0,-1j/2],
              [1j/2,0]])
sz=np.matrix([[1/2,0],
              [0,-1/2]])

idd=np.matrix([[1,0],
              [0,1]])

# S1=np.array([np.kron(sx,idd),
#              np.kron(sy,idd),
#              np.kron(sz,idd)])

# S2=np.array([np.kron(idd,sx),
#               np.kron(idd,sy),
#               np.kron(idd,sz)])
S1=np.array([np.kron(np.kron(np.kron(sx,idd),idd),idd),
              np.kron(np.kron(np.kron(sy,idd),idd),idd),
              np.kron(np.kron(np.kron(sz,idd),idd),idd)])



S2=np.array([np.kron(np.kron(np.kron(idd,sx),idd),idd),
              np.kron(np.kron(np.kron(idd,sy),idd),idd),
              np.kron(np.kron(np.kron(idd,sz),idd),idd)])

R2=[0.82915, 0.42085, 0]
R3=[0.67085, 0.57915, 0]

R1=[0.42085, 0.67085, 0]
R4=[0.57915, 0.82915, 0]
A=0.9
C=0.0
B=0.9
#-------------Parallel h---------------

ptnum=81 #change this to chagne point numbers to be binned on hk
n=ptnum
# step=(2-(-2))/(81-1)

workloads=[n//proc_size for i in range(proc_size)]
for i in range(n % proc_size):
    workloads[i] += 1

start=0

for i in range(my_rank):
    start=start+workloads[i]
end=start+workloads[my_rank]
# print(start,end)

v=3
H=[1,-1,0]
H=H/np.linalg.norm(H)
H=2*H
Q_powder=[]
I_powder=[]
h=np.linspace(-5,5,ptnum)
k=np.linspace(-5,5,ptnum)
l=np.linspace(-7,7,41)
counter=0
for x in range(start, end):
    for y in range(len(k)):
        for z in range(len(l)):            
            Q=np.array([h[x],k[y],l[z]]) @ np.array([b1,b2,b3])
            I_powder.append((SQ(h[x],k[y],l[z],v,R2,R3, A, C, B,H)+SQ(h[x],k[y],l[z],v,R1,R4, A, -C, B,H)).real)
            Q_powder.append(np.linalg.norm(Q))
            counter=counter+1
# print(h[start,end])
Q_powder=np.array(Q_powder)
I_powder=np.array(I_powder)

# print(Q_powder)

    
if my_rank == 0:
    Q_par=Q_powder
    I_par=I_powder
    for source in range(1, proc_size):
        message=world_comm.recv(source=source)
        Q_par=np.append(Q_par,message[0])
        I_par=np.append(I_par,message[1])
    Q_s=np.sort(Q_par)
    I_s=I_par[np.argsort(Q_par)]
    Int,Q,_= binned_statistic(Q_s, I_s,statistic='mean', bins=50)
    Qplot=[]
    for i in range(len(Q)-1):
        Qplot.append( (Q[i]+Q[i+1])/2 )
    plt.plot(Qplot,Int,'.')
    plt.savefig(r'C:\Users\qmc\OneDrive\ONRL\Data\level%iwithfield.png'%v, format='png',dpi=400)
    print("Figure saved")
    save=np.transpose([Qplot,Int])
    # np.savetxt("QI level {i} Field {H}.txt" %v,save)
    print("Data Saved")
    
else:
    world_comm.send([Q_powder,I_powder], dest=0)

MPI.Finalize