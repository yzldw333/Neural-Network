import abc
import numpy as np
import random
class BaseLayer(metaclass  = abc.ABCMeta):

    last_layer = None
    next_layer = None
    value = None #n*m array for forward computation
    num = 0     #num of elements in column not num of samples in row
    delta = None #n*m array for backward computation
    name = None
    def forward_compute(self):
        pass
    def backward_compute(self):
        pass
    def SetLast(self,lastLayer):
        self.last_layer = lastLayer
    def SetNext(self,nextLayer):
        self.next_layer = nextLayer
    def storeParameters(self):   #钩子函数,有需要子类就进行实现
        pass

class ActivateLayer(BaseLayer):
    def SetLast(self,lastLayer):
        self.last_layer = lastLayer
        self.num = lastLayer.num

class InputLayer(BaseLayer):
    # for convolution NN
    images = None
    channel = None
    width = None
    height = None

    def __init__(self,channel,height,width):
        self.channel = channel
        self.width = width
        self.height = height
        self.num = channel*width*height

    #def __init__(self,num):
        #self.num = num


    def setValue(self,value):
        self.images = value
        m = np.size(self.images,0)
        self.value = self.images.reshape(m,self.channel*self.height*self.width)
        self.value = self.value.transpose()
        self.num = self.channel*self.height*self.width


class BaseConvolutionLayer(BaseLayer):
    channel = None
    squareSize = None
    width = None
    height = None
    images = None
    stride = None

    def __init__(self,squareSize,stride):
        self.squareSize = squareSize
        self.stride = stride

    def SetLast(self,lastLayer):
        super().SetLast(lastLayer)
        self.width = int((lastLayer.width-self.squareSize)/self.stride + 1)
        self.height = int((lastLayer.height-self.squareSize)/self.stride + 1)

    def unRollImages(self,index):
        i = index
        old_images = self.last_layer.images
        m,old_channel,old_width,old_height = old_images.shape
        newData = []
        #Process unroll the data
        for c in range(old_channel):
            tmp = []
            for h in range(0,old_height-self.squareSize+1,self.stride):
                for w in range(0,old_width-self.squareSize+1,self.stride):
                    tmp.append(old_images[i,c,h:h+self.squareSize,w:w+self.squareSize].reshape(1,self.squareSize**2))
            newData.append(tmp)
        newData = np.array(newData).reshape(old_channel,self.width*self.height,self.squareSize**2)
        newData = newData.transpose(0,2,1)
        return newData
    def unRollImagesForConv(self,index):
        '''对上一层第index张图片展开
            得到 新width*新height,   卷积核平方*旧channel数  这样二维size的上层的图片
        '''
        i = index
        old_images = self.last_layer.images
        m,old_channel,old_height,old_width = old_images.shape
        newData = []
        #Process unroll the data
        for h in range(0,old_height-self.squareSize+1,self.stride):
            for w in range(0,old_width-self.squareSize+1,self.stride):
                tmp = []
                for c in range(old_channel):
                    tmp.append(old_images[i,c,h:h+self.squareSize,w:w+self.squareSize].reshape(1,self.squareSize**2))
                    #h,w像素位置的,一个个通道添加到列表,得到old_channel   *   squaireSize平方的形式
                tmp = np.array(tmp).reshape(1,self.squareSize**2*old_channel)
                #对此点进行reshape,把形式变成一整行,并添加到newData
                newData.append(tmp)
                #这里就相当于把卷积后的像素点所需区域,一行行的添加到列表
        newData = np.array(newData).reshape(self.width*self.height,self.squareSize**2*old_channel)
        return newData


class ConvolutionLayer(BaseConvolutionLayer):
    filters = None
    bias = None
    epsilon = 0.1
    grad_filters = None
    rate = 0.3
    grad_bias = None
    output_layer = None
    old_unroll_images_list = None
    def __init__(self,channel,squareSize,stride=1):
        super().__init__(squareSize,stride)
        self.channel = channel

    def rate_modify(self):
        if self.output_layer == None:
            p = self.next_layer
            while p.next_layer != None:
                p = p.next_layer
            self.output_layer = p
        costValue = self.output_layer.costValue
        if costValue>2:
            self.rate = 0.3
        elif costValue>1.5:
            self.rate = 0.2
        elif costValue >1:
            self.rate = 0.1
        elif costValue >0.5:
            self.rate = 0.03
        elif costValue >0.07:
            self.rate = 0.02
        elif costValue >0.04:
            self.rate = 0.015

    def initParameters(self):

        try:
            parameterFile = np.load(self.name+'.npz')
            self.filters = parameterFile['arr_0']
            self.bias = parameterFile['arr_1']
        except FileNotFoundError as err:
            self.filters = np.random.rand(self.squareSize**2*self.last_layer.channel,self.channel)*self.epsilon*2-self.epsilon
            self.bias = np.random.rand(self.width*self.height,self.channel)*self.epsilon*2-self.epsilon
    def storeParameters(self):
        np.savez(self.name+'.npz',self.filters,self.bias)


    def SetLast(self,lastLayer):
        super().SetLast(lastLayer)
        self.initParameters()
        self.num = self.channel*self.height*self.width
        pass

    def forward_compute(self):
        old_images = self.last_layer.images
        m,old_channel,old_width,old_height = old_images.shape
        result = []
        self.old_unroll_images_list = []
        for i in range(m):
            #compute the data
            newData = self.unRollImagesForConv(i)
            self.old_unroll_images_list.append(newData)
            convImage = newData.dot(self.filters)
            convImage+=self.bias
            convImage = convImage.transpose()
            result.append(convImage)
        #reshape the data
        self.images = np.array(result).reshape(m,self.channel,self.height,self.width)
        self.value = self.images.reshape(m,self.channel*self.height*self.width)
    def backward_compute(self):
        self.rate_modify()
        m = np.size(self.images,0)
        self.delta = self.delta.reshape(m,self.channel,self.height,self.width)
        filters_grad = np.zeros([self.squareSize**2*self.last_layer.channel,self.channel])
        bias_grad = np.zeros([self.width*self.height,self.channel])
        lastDelta = np.zeros([m,self.last_layer.channel,self.last_layer.height,self.last_layer.width])
        for i in range(m):
            oldImage = self.old_unroll_images_list[i]
            tmpDelta = self.delta[i,:,:,:].reshape(self.channel,self.height*self.width).transpose()
            newDelta = tmpDelta.dot(self.filters.transpose())       #format(width*height,  squareSize**2 * oldchannel)
            for c in range(self.last_layer.channel):
                for h in range(self.height):
                    for w in range(self.width):
                        lastW = w*self.stride
                        lastH = h*self.stride
                        startSquare = self.squareSize**2*c
                        tmpValue = newDelta[h*self.width+w,startSquare:startSquare+self.squareSize**2].reshape(self.squareSize,self.squareSize)
                        lastDelta[i,c,lastH:lastH+self.squareSize,lastW:lastW+self.squareSize] += tmpValue
            new_grad = oldImage.transpose().dot(tmpDelta)
            bias_grad += tmpDelta
            filters_grad+=new_grad
        filters_grad/=(1.0*m)
        bias_grad/=(1.0*m)
        self.grad_bias = bias_grad    #store in object
        self.bias-=bias_grad*self.rate
        self.grad_filters = filters_grad   #store in object
        self.filters-=filters_grad*self.rate
        self.last_layer.delta = lastDelta
        pass

class PoolingLayer(BaseConvolutionLayer):
    images = None
    channel = None
    squareSize = None
    stride = None
    width = None
    height = None
    def __init__(self,squareSize,stride):
        super().__init__(squareSize,stride)

    def SetLast(self,lastLayer):
        super().SetLast(lastLayer)
        self.channel = lastLayer.channel
        self.num = self.channel*self.height*self.width

class MaxPoolingLayer(PoolingLayer):
    maxIndex = None
    def __init__(self,squareSize,stride):
        super().__init__(squareSize,stride)
    def SetLast(self,lastLayer):
        super().SetLast(lastLayer)
    def forward_compute(self):
        old_images = self.last_layer.images
        m,old_channel,old_width,old_height = old_images.shape
        result = []
        self.maxIndex = []
        for i in range(m):
            newData = self.unRollImages(i)
            #compute the data
            self.maxIndex.append(list(np.argmax(newData,1)))
            result.append(list(np.max(newData,1)))
        self.maxIndex = np.array(self.maxIndex).reshape(m,self.channel,self.width*self.height)
        #reshape the data
        self.images = np.array(result).reshape(m,self.channel,self.height,self.width)
        self.value = self.images.reshape(m,self.channel*self.height*self.width)
    def backward_compute(self):
        m = np.size(self.images,0)
        self.delta = self.delta.reshape(m,self.channel,self.height,self.width)
        newDelta = np.zeros([m,self.last_layer.channel,self.last_layer.height,self.last_layer.width])
        for i in range(m):
            for j in range(self.channel):
                for h in range(self.height):
                    for w in range(self.width):
                        tmpLoc = self.maxIndex[i,j,h*self.width+w]
                        relativeH = tmpLoc//self.squareSize
                        relativeW = tmpLoc - relativeH * self.squareSize
                        lastW = w*self.stride+relativeW
                        lastH = h*self.stride+relativeH
                        newDelta[i,j,lastH,lastW] += self.delta[i,j,h,w]
        self.last_layer.delta = newDelta
        pass

class AvgPoolingLayer(PoolingLayer):
    w = None
    bias = None
    old_unroll_images_list = []
    grad_w = None
    rate = 0.1
    def __init__(self,squareSize,stride):
        super().__init__(squareSize,stride)
    def SetLast(self,lastLayer):
        super().SetLast(lastLayer)
        self.w = np.ones([self.channel,1])*1.0/self.squareSize**2
        self.bias = np.random.rand(self.channel,1)

    def forward_compute(self):
        old_images = self.last_layer.images
        m,old_channel,old_width,old_height = old_images.shape
        result = []
        self.old_unroll_images_list = []
        for i in range(m):
            newData = self.unRollImages(i)
            self.old_unroll_images_list.append(newData)
            #compute the data
            computeSum = np.sum(newData,1).reshape(self.channel,self.height*self.width)
            result.append(list(computeSum*self.w+self.bias))
        #reshape the data
        self.images = np.array(result).reshape(m,self.channel,self.height,self.width)
        self.value = self.images.reshape(m,self.channel*self.height*self.width)

    def backward_compute(self):
        m,old_channel,old_width,old_height = self.last_layer.images.shape
        oldDelta = np.zeros([m,old_channel,old_height,old_width])
        w_grad = np.zeros([self.channel,1])
        bias_grad = np.zeros([self.channel,1])
        for i in range(m):
            tmpDelta = self.delta[i,:].reshape(self.channel,self.height*self.width)
            old_unroll_image = self.old_unroll_images_list[i]
            computeSum = np.sum(old_unroll_image,1).reshape(self.channel,self.height*self.width)
            for c in range(self.channel):
                for h in range(self.height):
                    for w in range(self.width):
                        lastW = w*self.stride
                        lastH = h*self.stride
                        tmpValue = tmpDelta[c,h*self.width+w]
                        oldDelta[i,c,lastH:lastH+self.squareSize,lastW:lastW+self.squareSize] += tmpValue*self.w[c,0]
                        w_grad[c,0]+=computeSum[c,h*self.width+w]*tmpValue
                        bias_grad[c,0]+=tmpValue
        w_grad/=(1.0*m)
        bias_grad/=(1.0*m)
        self.grad_w = w_grad
        self.w -= self.grad_w*self.rate
        self.bias -= bias_grad*self.rate
        self.last_layer.delta = oldDelta


class FullCrossLayer(BaseLayer):
    theta = None
    grad_theta = None
    epsilon = 0.1
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
        if costValue>2:
            self.rate = 0.3
        elif costValue>1.5:
            self.rate = 0.2
        elif costValue >1:
            self.rate = 0.1
        elif costValue >0.5:
            self.rate = 0.03
        elif costValue >0.07:
            self.rate = 0.02
        elif costValue >0.04:
            self.rate = 0.015

    def initParameters(self):

        try:
            parameterFile = np.load(self.name+'.npz')
            self.theta = parameterFile['arr_0']
        except FileNotFoundError as err:
            self.theta = np.random.rand(self.last_layer.num+1,self.num)*2*self.epsilon-self.epsilon




    def storeParameters(self):
        np.savez(self.name+'.npz',self.theta)

    def SetLast(self,lastLayer):
        super().SetLast(lastLayer)
        self.initParameters()
        pass
    def forward_compute(self):
        m = np.size(self.last_layer.value,0)
        tmp = np.hstack((np.ones([m,1]),self.last_layer.value))
        self.value = tmp.dot(self.theta)
    def backward_compute(self):
        self.rate_modify()
        m = np.size(self.last_layer.value,0)
        #self.delta = self.next_layer.theta.transpose().dot(self.next_layer.delta)
        tmp = np.hstack((np.ones([m,1]),self.last_layer.value))
        self.last_layer.delta = self.delta.dot(self.theta.transpose())
        self.last_layer.delta = self.last_layer.delta[:,1:]
        self.grad_theta = tmp.transpose().dot(self.delta)/(1.0*m)
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
    lamb = 0
    costFunc = None
    costValue = None
    def __init__(self):
        pass
    def LMS(self):
        res = np.sum((self.h-self.y)**2)/(2.0*np.size(self.y,0))
        self.costValue = res
        return res
    def SoftMax(self):
        self.costValue = -np.sum(self.y*np.log(self.h))/(1.0*np.size(self.y,0))
        return self.costValue


    def init(self,costFuncName='LMS'):
        if costFuncName is 'LMS':
            self.costFunc = self.LMS
            return True
        elif costFuncName is 'SoftMax':
            self.costFunc = self.SoftMax
            return True
        return False
    def setY(self,y):
        self.y = y
    def forward_compute(self):
        if self.costFunc == self.LMS:
            self.h = self.last_layer.value
        elif self.costFunc == self.SoftMax:
            m = np.size(self.last_layer.value,0)
            fenmu = np.sum(np.exp(self.last_layer.value),axis=1).reshape(m,1)
            self.h = np.exp(self.last_layer.value)/fenmu
        self.costValue=self.costFunc()
    def backward_compute(self):
        self.last_layer.delta = self.h-self.y






