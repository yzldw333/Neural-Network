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
    initLayerList = {'INPUT_LAYER':False,'OUTPUT_LAYER':False}
    initializeList = {'CONNECT_LAYERS':False}
    name = None
    def __init__(self,name):
        self.name = name
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

        # index = 1
        # n = np.size(self.convolution_pooling_layer[index].w,0)
        # m = np.size(self.convolution_pooling_layer[index].w,1)
        # grad_list = []
        # for i in range(n):
        #     for j in range(m):
        #         self.convolution_pooling_layer[index].w[i,j]+=epsilon
        #         self.forward_compute()
        #         Ju = self.costFunction()
        #         self.convolution_pooling_layer[index].w[i,j]-=epsilon*2
        #         self.forward_compute()
        #         Jd = self.costFunction()
        #         grad_list.append((Ju-Jd)/(2*epsilon))
        #         self.convolution_pooling_layer[index].w[i,j]+=epsilon
        # grad_list = np.array(grad_list)
        # self.forward_compute()
        # self.backward_compute()
        # bp_grad_list = self.convolution_pooling_layer[index].grad_w.ravel()
        # distance = np.sum((grad_list-bp_grad_list)**2)/len(bp_grad_list)
        # print(distance)
        # if distance<10e-6:
        #     print('AvgPooling Layer Gradiant check pass!')
        # else:
        #     print('AvgPooling Layer Gradiant check failed!')
        #     return False

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


    def SetInputLayer(self,channel,width=1,height=1):
        self.input_layer = InputLayer(channel,width,height)
        self.initLayerList['INPUT_LAYER'] = True

    def SetOutputLayer(self,costFuncName):
        self.output_layer = OutputLayer()
        self.initLayerList['OUTPUT_LAYER'] = self.output_layer.init(costFuncName)

    def SetLayerSequences(self,LayerDictionary):
        '''
            You need to set input and output layer first, then use this function to set other layers between them.
        :param LayerDictionary: LayerDictionary format is like this.
                                ('ConvolutionLayer',(channel,height,width))
                                ('MaxPoolingLayer',(squareHeight,squareWidth))
                                ('AvgPoolingLayer',(squareHeight,squareWidth))
                                ('FullCrossLayer',num)
                                ('Sigmoid',None)
                                ('Tanh',None)
                                ('ReLu',None)
        :return: True or False.
        '''
        if self.initLayerList['INPUT_LAYER'] == False or self.initLayerList['OUTPUT_LAYER'] == False:
            print('Input layer or output layer has not been initialized.')
            return False
        for (name,setting) in LayerDictionary:
            newLayer = None
            if name == 'ConvolutionLayer':
                newLayer = ConvolutionLayer(setting[0],setting[1],setting[2])
                newLayer.name = self.name+str(len(self.sequences))+'layer'
                self.convolution_pooling_layer.append(newLayer)
            elif name == 'MaxPoolingLayer':
                newLayer = MaxPoolingLayer(setting[0],setting[1])
            elif name == 'AvgPoolingLayer':
                newLayer = AvgPoolingLayer(setting[0],setting[1])
            elif name == 'FullCrossLayer':
                newLayer = FullCrossLayer(setting)  # setting is a integer which means num of FC Layer
                newLayer.name = self.name+str(len(self.sequences))+'layer'
                self.full_cross_layer.append(newLayer)
            elif name == 'Sigmoid':
                newLayer = SigmoidLayer()
            elif name == 'Tanh':
                newLayer = TanhLayer()
            elif name == 'ReLu':
                newLayer = ReLuLayer()
            else:
                continue
            if len(self.sequences)!= 0:
                newLayer.SetLast(self.sequences[-1])
                self.sequences[-1].SetNext(newLayer)
            else:
                newLayer.SetLast(self.input_layer)
            self.sequences.append(newLayer)
        if len(self.sequences)!= 0:
            self.output_layer.SetLast(self.sequences[-1])
            self.sequences[-1].SetNext(self.output_layer)
            self.sequences.append(self.output_layer)
        else:
            self.output_layer.SetLast(self.input_layer)
            self.sequences.append(self.output_layer)
        return True

    def SetConvolutionPoolingLayer(self,LayerDictionary):
        pass



    def SetFullConnectLayer(self,LayerDictionary):
        pass

    # def SetActivationLayer(self,activeName):
    #     if activeName == 'Sigmoid':
    #         self.activeName = 'Sigmoid'
    #     elif activeName == 'Tanh':
    #         self.activeName = 'Tanh'
    #     elif activeName == 'ReLu':
    #         self.activeName = 'ReLu'
    #     self.initLayerList['ACTIVATION_LAYER'] = True


    def ConnectLayers(self):
        pass


    def SetTrainSamples(self,x,y):
        self.train_x = x
        self.train_y = y
    def SetTestSamples(self,x,y):
        self.test_x = x
        self.test_y = y
    def forward_compute(self):
        for layer in self.sequences:
            layer.forward_compute()

    def backward_compute(self):
        for layer in self.sequences[::-1]:
            layer.backward_compute()

    def costFunction(self):
        return self.output_layer.costFunc()

    def minibatch_train(self,batch_size,steps,ifshow=True,gradiantCheck=True):

        if self.safety_check(gradiantCheck) is False:
            return False
        start_time = datetime.datetime.now()
        tmp_time = start_time
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
        startid = 0
        for i in range(steps):
            #recycle choose
            if startid >=total_size:
                startid%=total_size
            endid = startid+tmp_size
            choose = []
            if endid >=total_size:
                endid%=total_size
                choose+=list(range(startid,total_size))
                choose+=list(range(0,endid))
            else:
                choose+=list(range(startid,endid))
            startid+=tmp_size
            # random choose
            # if i%5==0:
            #     choose = random.sample(range(total_size),tmp_size)
            #choose = random.sample(range(total_size),tmp_size)
            self.input_layer.setValue(self.train_x[choose,:])

            self.output_layer.setY(self.train_y[choose,:])
            self.forward_compute()
            self.backward_compute()
            if np.isnan(self.output_layer.costValue):
                print('Nan, quit the program!')
                return False
            draw_cost.append(self.output_layer.costValue)
            draw_steps.append(i)
            new_tmp_time = datetime.datetime.now()
            if ifshow:
                print('Step %s complete! CostValue %s  StepTime %s'%(i,self.output_layer.costValue,new_tmp_time-tmp_time))
            tmp_time = new_tmp_time
        end_time = datetime.datetime.now()
        print('Time:%s'%(end_time))
        print('BP Cost Time:%s'%(end_time-start_time))
        self.store_parameters()
        print('Save Parameters...')
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
            self.forward_compute()
            if self.output_layer.costFunc == self.output_layer.LMS:
                if self.output_layer.costFunc()<0.1:
                    right+=1
            if self.output_layer.costFunc == self.output_layer.SoftMax:
                if self.output_layer.y[0,self.output_layer.h.argmax()]==1:
                    right+=1

        print('Correct percentage: %s'%(right*1.0/testNum*100))















