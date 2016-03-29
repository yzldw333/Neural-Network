import numpy as np
import random
from NeuralNetwork.Layers.core import *
import matplotlib.pyplot as plt
import datetime
class NeuralModel:
    sequences = []
    input_layer = None
    output_layer = None
    activeName = None

    full_cross_layer = []
    activation_layer = []
    train_x = None
    train_y = None
    test_x = None
    test_y = None
    initLayerList = {'INPUT_LAYER':False,'OUTPUT_LAYER':False,'FULL_CONNECT_LAYER':False}
    initializeList = {'CONNECT_LAYERS':False}

    def gradiant_check(self):
        'just check the first inner product layer'
        linearList = []
        epsilon = 0.0001
        self.input_layer.setValue(self.train_x[:,:2])
        self.output_layer.setY(self.train_y[:,:2])
        n = np.size(self.full_cross_layer[0].theta,0)
        m = np.size(self.full_cross_layer[0].theta,1)
        grad_list = []
        for i in range(n):
            for j in range(m):
                self.full_cross_layer[0].theta[i,j]+=epsilon
                self.forward_compute()
                Ju = self.costFunction()
                self.full_cross_layer[0].theta[i,j]-=epsilon*2
                self.forward_compute()
                Jd = self.costFunction()
                grad_list.append((Ju-Jd)/(2*epsilon))
                self.full_cross_layer[0].theta[i,j]+=epsilon
        grad_list = np.array(grad_list)
        self.forward_compute()
        self.backward_compute()
        bp_grad_list = self.full_cross_layer[0].grad_theta.ravel()
        distance = np.sum((grad_list-bp_grad_list)**2)/len(bp_grad_list)
        print(distance)
        if distance<10e-6:
            print('Gradiant check pass!')
            return True
        else:
            print('Gradiant check failed!')
            return False


    def safety_check(self):
        #step 1
        res = True
        for e in self.initLayerList:
            if self.initLayerList[e] is False:
                print('%s has not been initialized.'%e)
                res = False
        if res is False:
            return res

        #step 2
        if self.initializeList['CONNECT_LAYERS']is False:
            print('LAYERS has not been connected.')
            print('Connect them now...')
            self.initializeList['CONNECT_LAYERS'] = self.ConnectLayers()
            print('LAYERS has been connected.')

        #step 3
        if self.gradiant_check() is False:
            return False
        print('Safety check pass!')
        return True

    def SetInputLayer(self,inputNum):
        self.input_layer = InputLayer(inputNum)
        self.initLayerList['INPUT_LAYER'] = True

    def SetOutputLayer(self,costFuncName):
        self.output_layer = OutputLayer()
        self.output_layer.init(costFuncName)
        self.initLayerList['OUTPUT_LAYER'] = True




    def SetFullConnectLayer(self,LayerDictionary):
        for (num,activeFunc) in LayerDictionary:
            self.full_cross_layer.append(FullCrossLayer(num))
            if activeFunc == 'Sigmoid':
                self.full_cross_layer.append(SigmoidLayer())
            elif activeFunc== 'Tanh':
                self.full_cross_layer.append(TanhLayer())
            elif activeFunc== 'ReLu':
                self.full_cross_layer.append(ReLuLayer())
        self.initLayerList['FULL_CONNECT_LAYER'] = True

    # def SetActivationLayer(self,activeName):
    #     if activeName == 'Sigmoid':
    #         self.activeName = 'Sigmoid'
    #     elif activeName == 'Tanh':
    #         self.activeName = 'Tanh'
    #     elif activeName == 'ReLu':
    #         self.activeName = 'ReLu'
    #     self.initLayerList['ACTIVATION_LAYER'] = True


    def ConnectLayers(self):
        self.sequences = []
        self.sequences.append(self.input_layer)
        for layer in self.full_cross_layer:
            self.sequences[-1].SetNext(layer)
            layer.SetLast(self.sequences[-1])
            self.sequences.append(layer)
        self.sequences[-1].SetNext(self.output_layer)
        self.output_layer.SetLast(self.sequences[-1])
        self.sequences.append(self.output_layer)
        return True








    def SetTrainSamples(self,x,y):
        self.train_x = x
        self.train_y = y
    def SetTestSamples(self,x,y):
        self.test_x = x
        self.test_y = y
    def forward_compute(self):
        layerNum = len(self.sequences)
        for i in range(1,layerNum):
            self.sequences[i].forward_compute()

    def backward_compute(self):
        layerNum = len(self.sequences)
        for j in range(1,layerNum)[::-1]:
            self.sequences[j].backward_compute()

    def costFunction(self):
        return self.output_layer.costFunc()

    def minibatch_train(self,batch_size,steps,ifshow):

        if self.safety_check() is False:
            return False
        start_time = datetime.datetime.now()
        print('Time:%s\t\t\tStart computing.\n'%start_time)
        total_size = np.size(self.train_x,1)
        if batch_size > total_size:
            print('batch size is too large.')
            return False
        elif batch_size<=1:
            print('batch size is too small.')
            return False
        draw_cost = []
        draw_steps = []
        for i in range(steps):
            choose = random.sample(range(total_size),batch_size)
            self.input_layer.setValue(self.train_x[:,choose])
            self.output_layer.setY(self.train_y[:,choose])
            self.forward_compute()
            self.backward_compute()
            draw_cost.append(self.output_layer.costFunc())
            draw_steps.append(i)
            if ifshow:
                print('Step %s complete!'%i)
        end_time = datetime.datetime.now()
        print('Time:%s'%(end_time))
        print('BP Cost Time:%s'%(end_time-start_time))
        plt.plot(draw_steps,draw_cost,marker='.')
        plt.title('Train with batch size %s' % batch_size)
        plt.show()
        return True

    def batch_train(self,steps):
        total_size = np.size(self.train_x,1)
        self.minibatch_train(total_size,steps)

    def test_error(self):
        right = 0
        layerNum = len(self.sequences)
        testNum = np.size(self.test_x,1)
        inputSize = np.size(self.test_x,0)
        outputSize = np.size(self.test_y,0)
        for i in range(testNum):
            testx = self.test_x[:,i].reshape(inputSize,1)
            testy = self.test_y[:,i].reshape(outputSize,1)
            self.input_layer.setValue(testx)
            self.output_layer.setY(testy)
            for j in range(1,layerNum):
                self.sequences[j].forward_compute()
            if self.output_layer.costFunc == self.output_layer.LMS:
                if self.output_layer.costFunc()<0.1:
                    right+=1
            if self.output_layer.costFunc == self.output_layer.SoftMax:
                if self.output_layer.y[self.output_layer.h.argmax()]==1:
                    right+=1

        print('Correct percentage: %s'%(right*1.0/testNum*100))















