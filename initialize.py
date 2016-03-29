import numpy
from numpy import *
import matplotlib.pyplot as plt
def initialize(radius,width,D,na,nb):
    r = radius
    w = width
    d = D
    RW = r+w/2.0
    Rw = r-w/2.0
    Na = na
    Nb = nb
    X = zeros([Na+Nb,3]) #will be shuffled
    #Random initialization
    #UP Half Circle
    for i in range(Na):
        rangeX = 0
        rangeY = 0
        while True:
            rangeX = random.random()*RW*2-RW
            rangeY = random.random()*RW
            if Rw**2<rangeX**2+rangeY**2<RW**2:
                break
        X[i,0] = rangeX
        X[i,1] = rangeY
        X[i,2] = 1
    #Down Half Circle
    for i in range(Nb):
        rangeX = 0
        rangeY = 0
        while True:
            rangeX = random.random()*RW*2-w/2
            rangeY = random.random()*(-RW)-d
            # center (r,-d)
            if Rw**2<(rangeX-r)**2+(rangeY+d)**2<RW**2:
                break
        X[i+Na,0] = rangeX
        X[i+Na,1] = rangeY
        X[i+Na,2] = -1
    random.shuffle(X)
    X = X.transpose()
    x = X[0:2,:]
    y = X[2,:].reshape(1,Na+Nb)
    # plt.plot(X1[:,1],X1[:,2],marker='+',linewidth=0)
    # plt.plot(X2[:,1],X2[:,2],marker='+',linewidth=0)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    return (x,y)
    #plt.plot(X2[:,1],X2[:,2])

