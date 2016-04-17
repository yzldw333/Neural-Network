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
    convolution_pooling_layer = []
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
        epsilon = 0.000001
        self.input_layer.setValue(self.train_x[:2,:])
        self.output_layer.setY(self.train_y[:2,:])
        # index = 0
        # n = np.size(self.full_cross_layer[index].theta,0)
        # m = np.size(self.full_cross_layer[index].theta,1)
        # grad_list = []
        # for i in range(n):
        #     for j in range(m):
        #         self.full_cross_layer[index].theta[i,j]+=epsilon
        #         self.forward_compute()
        #         Ju = self.costFunction()
        #         self.full_cross_layer[index].theta[i,j]-=epsilon*2
        #         self.forward_compute()
        #         Jd = self.costFunction()
        #         grad_list.append((Ju-Jd)/(2*epsilon))
        #         self.full_cross_layer[index].theta[i,j]+=epsilon
        # grad_list = np.array(grad_list)
        # self.forward_compute()
        # self.backward_compute()
        # bp_grad_list = self.full_cross_layer[index].grad_theta.ravel()
        # distance = np.sum((grad_list-bp_grad_list)**2)/len(bp_grad_list)
        # print(distance)
        # if distance<10e-6:
        #     print('FC Layer Gradiant check pass!')
        # else:
        #     print('FC Layer Gradiant check failed!')
        #     return False
        index = 1
        n = np.size(self.convolution_pooling_layer[index].w,0)
        m = np.size(self.convolution_pooling_layer[index].w,1)
        grad_list = []
        for i in range(n):
            for j in range(m):
                self.convolution_pooling_layer[index].w[i,j]+=epsilon
                self.forward_compute()
                Ju = self.costFunction()
                self.convolution_pooling_layer[index].w[i,j]-=epsilon*2
                self.forward_compute()
                Jd = self.costFunction()
                grad_list.append((Ju-Jd)/(2*epsilon))
                self.convolution_pooling_layer[index].w[i,j]+=epsilon
        grad_list = np.array(grad_list)
        self.forward_compute()
        self.backward_compute()
        bp_grad_list = self.convolution_pooling_layer[index].grad_w.ravel()
        distance = np.sum((grad_list-bp_grad_list)**2)/len(bp_grad_list)
        print(distance)
        if distance<10e-6:
            print('AvgPooling Layer Gradiant check pass!')
        else:
            print('AvgPooling Layer Gradiant check failed!')
            return False

        index = 0
        n = np.size(self.convolution_pooling_layer[index].filters,0)
        m = np.size(self.convolution_pooling_layer[index].filters,1)
        grad_list = []
        for i in range(n):
            for j in range(m):
                self.convolution_pooling_layer[index].filters[i,j]+=epsilon
                self.forward_compute()
                Ju = self.costFunction()
                self.convolution_pooling_layer[index].filters[i,j]-=epsilon*2
                self.forward_compute()
                Jd = self.costFunction()
                grad_list.append((Ju-Jd)/(2*epsilon))
                self.convolution_pooling_layer[index].filters[i,j]+=epsilon
        grad_list = np.array(grad_list)
        self.forward_compute()
        self.backward_compute()
        bp_grad_list = self.convolution_pooling_layer[index].grad_filters.ravel()
        distance = np.sum((grad_list-bp_grad_list)**2)/len(bp_grad_list)
        print(distance)
        if distance<10e-6:
            print('Conv Layer filter Gradiant check pass!')
        else:
            print('Conv Layer Gradiant check failed!')
            return False

        index = 0
        n = np.size(self.convolution_pooling_layer[index].bias,0)
        m = np.size(self.convolution_pooling_layer[index].bias,1)
        grad_list = []
        for i in range(n):
            for j in range(m):
                self.convolution_pooling_layer[index].bias[i,j]+=epsilon
                self.forward_compute()
                Ju = self.costFunction()
                self.convolution_pooling_layer[index].bias[i,j]-=epsilon*2
                self.forward_compute()
                Jd = self.costFunction()
                grad_list.append((Ju-Jd)/(2*epsilon))
                self.convolution_pooling_layer[index].bias[i,j]+=epsilon
        grad_list = np.array(grad_list)
        self.forward_compute()
        self.backward_compute()
        bp_grad_list = self.convolution_pooling_layer[index].grad_bias.ravel()
        distance = np.sum((grad_list-bp_grad_list)**2)/len(bp_grad_list)
        print(distance)
        if distance<10e-6:
            print('Conv Layer bias Gradiant check pass!')
        else:
            print('Conv Layer bias Gradiant check failed!')
            return False



    def safety_check(self,gradientCheck):
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
        if gradientCheck is True:
            if self.gradiant_check() is False:
                return False
            else:
                print('Safety check pass!')
        return True


    def SetInputLayer(self,channel,width,height):
        self.input_layer = InputLayer(channel,width,height)
        self.initLayerList['INPUT_LAYER'] = True

    def SetOutputLayer(self,costFuncName):
        self.output_layer = OutputLayer()
        self.initLayerList['OUTPUT_LAYER'] = self.output_layer.init(costFuncName)


    def SetConvolutionPoolingLayer(self,LayerDictionary):
        for (name,setting) in LayerDictionary:
            if name is 'ConvolutionLayer':
                self.convolution_pooling_layer.append(ConvolutionLayer(setting[0],setting[1],setting[2]))
            elif name is 'MaxPoolingLayer':
                self.convolution_pooling_layer.append(MaxPoolingLayer(setting[0],setting[1]))
            elif name is 'AvgPoolingLayer':
                self.convolution_pooling_layer.append(AvgPoolingLayer(setting[0],setting[1]))

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

        for layer in self.convolution_pooling_layer:
            self.sequences[-1].SetNext(layer)
            layer.name = str(len(self.sequences))+'layer'
            layer.SetLast(self.sequences[-1])
            self.sequences.append(layer)
        for layer in self.full_cross_layer:
            self.sequences[-1].SetNext(layer)
            layer.name = str(len(self.sequences))+'layer'
            layer.SetLast(self.sequences[-1])
            self.sequences.append(layer)
        self.sequences[-1].SetNext(self.output_layer)
        self.output_layer.SetLast(self.sequences[-1])
        self.sequences.append(self.output_layer)


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

    def minibatch_train(self,batch_size,steps,ifshow=True,gradiantCheck=True):

        if self.safety_check(gradiantCheck) is False:
            return False
        start_time = datetime.datetime.now()
        print('Time:%s\t\t\tStart computing.\n'%start_time)
        total_size = np.size(self.train_x,0)
        if batch_size > total_size:
            print('batch size is too large.')
            return False
        elif batch_size<=1:
            print('batch size is too small.')
            return False
        tmp_size = batch_size
        draw_cost = []
        draw_steps = []
        for i in range(steps):
            # if i%10==0:
            #     choose = random.sample(range(total_size),tmp_size)
            choose = random.sample(range(total_size),tmp_size)
            self.input_layer.setValue(self.train_x[choose,:])

            self.output_layer.setY(self.train_y[choose,:])
            self.forward_compute()
            self.backward_compute()
            draw_cost.append(self.output_layer.costValue)
            draw_steps.append(i)
            # if i%30 is 0:
            #     print(self.convolution_pooling_layer[0].filters)
            #     print(self.full_cross_layer[0].theta)
            if ifshow:
                print('Step %s complete! CostValue %s'%(i,self.output_layer.costValue))
            # if self.output_layer.costValue<0.2:
            #     tmp_size = batch_size*2
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

    def store_parameters(self):
        for layer in self.full_cross_layer:
            layer.storeParameters()
        for layer in self.convolution_pooling_layer:
            layer.storeParameters()
    def test_error(self):
        right = 0
        layerNum = len(self.sequences)
        testNum,channel,height,width = self.test_x.shape
        testNum,classNum = self.test_y.shape
        testNum = int(testNum*0.3)
        for i in range(testNum):

            testx = self.test_x[i,:].reshape(1,channel,height,width)
            testy = self.test_y[i,:].reshape(1,classNum)
            self.input_layer.setValue(testx)
            self.output_layer.setY(testy)
            for j in range(1,layerNum):
                self.sequences[j].forward_compute()
            if self.output_layer.costFunc == self.output_layer.LMS:
                if self.output_layer.costFunc()<0.1:
                    right+=1
            if self.output_layer.costFunc == self.output_layer.SoftMax:
                if self.output_layer.y[0,self.output_layer.h.argmax()]==1:
                    right+=1

        print('Correct percentage: %s'%(right*1.0/testNum*100))















