import abc
import numpy as np

class BaseLayer(metaclass  = abc.ABCMeta):

    last_layer = None
    next_layer = None
    value = None #n*m array for forward computation
    num = 0     #num of elements in column not num of samples in row
    delta = None #n*m array for backward computation
    def forward_compute(self):
        pass
    def backward_compute(self):
        pass
    def SetLast(self,lastLayer):
        self.last_layer = lastLayer
    def SetNext(self,nextLayer):
        self.next_layer = nextLayer

class ActivateLayer(BaseLayer):
    def SetLast(self,lastLayer):
        self.last_layer = lastLayer
        self.num = lastLayer.num

class InputLayer(BaseLayer):
    def __init__(self,num):
        self.num = num
        pass
    def setValue(self,value):
        self.value = value

class FullCrossLayer(BaseLayer):
    theta = None
    grad_theta = None
    epsilon = 0.2
    rate = 0.3
    output_layer = None
    def __init__(self,num):
        self.num = num
        pass
    def rate_modify(self):
        if self.output_layer == None:
            p = self.next_layer
            while p.next_layer != None:
                p = p.next_layer
            self.output_layer = p
        costValue = self.output_layer.costValue
        if costValue>0.3:
            self.rate = 0.3
        elif costValue >0.1:
            self.rate = 0.3
        elif costValue >0.08:
            self.rate = 0.2
        elif costValue >0.05:
            self.rate = 0.1
        elif costValue >0.03:
            self.rate = 0.07

    def SetLast(self,lastLayer):
        super().SetLast(lastLayer)
        self.theta = np.random.rand(self.num,lastLayer.num+1)*2*self.epsilon-self.epsilon
        pass
    def forward_compute(self):
        m = np.size(self.last_layer.value,1)
        tmp = np.vstack((np.ones([1,m]),self.last_layer.value))
        self.value = self.theta.dot(tmp)
    def backward_compute(self):
        self.rate_modify()
        m = np.size(self.last_layer.value,1)
        #self.delta = self.next_layer.theta.transpose().dot(self.next_layer.delta)
        tmp = np.vstack((np.ones([1,m]),self.last_layer.value))
        self.last_layer.delta = self.theta.transpose().dot(self.delta)
        self.last_layer.delta = self.last_layer.delta[1:,:]
        self.grad_theta = self.delta.dot(tmp.transpose())/(1.0*m)
        self.theta -= self.grad_theta*self.rate


class SigmoidLayer(ActivateLayer):
    def __init__(self):
        pass
    def forward_compute(self):
        self.value = 1.0/(1+np.exp(-self.last_layer.value))
    def backward_compute(self):
        self.last_layer.delta = self.delta*    (self.value)*(1-self.value)
        pass

class TanhLayer(ActivateLayer):
    def __init__(self):
        pass
    def forward_compute(self):
        self.value = np.tanh(self.last_layer.value)
    def backward_compute(self):
        self.last_layer.delta = self.delta*    (1-self.value**2)

class ReLuLayer(ActivateLayer):
    alpha = 1
    def __init__(self):
        pass
    def forward_compute(self):
        self.value = np.maximum(0,self.alpha*self.last_layer.value)
    def backward_compute(self):
        tmp = self.value.copy()
        tmp[tmp<=0] = 0
        tmp[tmp>0] = self.alpha
        self.last_layer.delta = self.delta*tmp

class OutputLayer(BaseLayer):
    h = None #hippothesis
    y = None #standard output
    costFunc = None
    costValue = None
    def __init__(self):
        pass
    def LMS(self):
        res = np.sum((self.h-self.y)**2)/(2.0*np.size(self.y,1))
        self.costValue = res
        return res
    def SoftMax(self):
        self.costValue = -np.sum(self.y*np.log(self.h))/(1.0*np.size(self.y,1))
        return self.costValue


    def init(self,costFuncName='LMS'):
        if costFuncName is 'LMS':
            self.costFunc = self.LMS
        elif costFuncName is 'SoftMax':
            self.costFunc = self.SoftMax
    def setY(self,y):
        self.y = y
    def forward_compute(self):
        if self.costFunc == self.LMS:
            self.h = self.last_layer.value
        elif self.costFunc == self.SoftMax:
            fenmu = np.sum(np.exp(self.last_layer.value),axis=0)
            self.h = np.exp(self.last_layer.value)/fenmu
    def backward_compute(self):
        self.last_layer.delta = self.h-self.y






