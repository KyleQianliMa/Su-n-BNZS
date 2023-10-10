import numpy as np

m0=np.array([1,-1,0])
m1=np.array([-1,1,0])

atom00=np.array([6.59842,  10.51812,  13.61310])
atom01=np.array([9.08038,  13.00008,  13.61310])

atom10=np.array([6.59842,   2.67872,  13.61310])
atom11=np.array([9.08038,   5.16068,  13.61310])

atom20=np.array([10.51812,   6.59842,   6.80655])
atom21=np.array([13.00008,   9.08038,   6.80655])

atom30=np.array([5.16068,   9.08038,   6.80655])
atom31=np.array([2.67872,   6.59842,   6.80655])

def H(atom0,atom1,m0,m1):
    r=atom0-atom1
    ur=r/np.linalg.norm(r)    
    H=(-1/(np.linalg.norm(r))**3) * (3*np.dot(m0,ur)*np.dot(m1,ur)-np.dot(m0,m1))
    return H

#---antiparallel----
H00_10=H(atom00,atom10,m0,m1)
H00_11=H(atom00,atom11,m0,m1)
H01_10=H(atom01,atom10,m0,m1)
H01_11=H(atom01,atom11,m0,m1)
H_inplane=H00_10+H00_11+H01_10+H01_11

H00_20=H(atom00,atom20,m0,m1)
H00_21=H(atom00,atom21,m0,m1)
H01_20=H(atom01,atom20,m0,m1)
H01_21=H(atom01,atom21,m0,m1)
H_outplane=H00_20+H00_21+H01_20+H01_21

H00_30=H(atom00,atom30,m0,m1)
H00_31=H(atom00,atom31,m0,m1)
H01_30=H(atom01,atom30,m0,m1)
H01_31=H(atom01,atom31,m0,m1)
H_neighbor=H00_30+H00_31+H01_30+H01_31

Htotal_anti=H_inplane+H_outplane+H_neighbor

#---parallel-----
P00_10=H(atom00,atom10,m0,m0)
P00_11=H(atom00,atom11,m0,m0)
P01_10=H(atom01,atom10,m0,m0)
P01_11=H(atom01,atom11,m0,m0)
P_inplane=P00_10+P00_11+P01_10+P01_11

P00_20=H(atom00,atom20,m0,m0)
P00_21=H(atom00,atom21,m0,m0)
P01_20=H(atom01,atom20,m0,m0)
P01_21=H(atom01,atom21,m0,m0)
P_outplane=P00_20+P00_21+P01_20+P01_21

P00_30=H(atom00,atom30,m0,m0)
P00_31=H(atom00,atom31,m0,m0)
P01_30=H(atom01,atom30,m0,m0)
P01_31=H(atom01,atom31,m0,m0)
P_neighbor=P00_30+P00_31+P01_30+P01_31

Ptotal_paral=P_inplane+P_outplane+P_neighbor

print(Htotal_anti, Ptotal_paral)
print(H_neighbor, P_neighbor)
print(H_inplane, P_inplane)