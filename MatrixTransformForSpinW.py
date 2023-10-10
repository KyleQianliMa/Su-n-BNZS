import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from sympy import symbols, solve,cos,sin,Matrix, Inverse


####################################################
  #Transformation for SSL matlab SpinW Operators
####################################################
g=np.matrix([[-0.0,   0.0,  0.0],
              [0.0 ,  0.0, 0.0],
              [0.0 ,  0.0, -0.2]])
def rotz(x):
    x=x/180 * np.pi
    Rc=np.matrix([[np.cos(x), -np.sin(x), 0],
                  [np.sin(x),np.cos(x),0],
                  [0, 0, 1]])
    return Rc

def roty(x):
    x=x/180 * np.pi
    Rc=np.matrix([[np.cos(x), 0, np.sin(x)],
                  [0,1,0],
                  [-np.sin(x), 0, np.cos(x)]])
    return Rc

def rotx(x):
    x=x/180 * np.pi
    Rc=np.matrix([[1, 0, 0],
                  [0,np.cos(x),-np.sin(x)],
                  [0,np.sin(x), np.cos(x)]])
    return Rc
gmatplus=np.linalg.inv(rotz(-45)) @ g @rotz(-45) #-45 for spinW, 45 for Sunny
print('For spinW:\n',gmatplus)
print('-----------')
#Use -45 for J1 in SpinW

def roty(x):
    x=x/180 * np.pi
    Rc=np.matrix([[np.cos(x), 0, np.sin(x)],
                  [0,1,0],
                  [-np.sin(x), 0, np.cos(x)]])
    return Rc
gy=np.linalg.inv(roty(45)) @ g @roty(0)
print('gy:\n',gy.round(2))
print('-----------')

gx=np.linalg.inv(rotx(180)) @ g @rotx(0)

print('gx:\n',gx.round(2))
print('-----------')