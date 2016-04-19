import numpy as np
from NeuralNetwork.Model.NeuralModel import NeuralModel
class NeuralNetwork:
    name = 'Neural Network'
    samples_x = None
    samples_y = None
    train_x = None
    train_y = None
    cvd_x = None
    cvd_y = None
    test_x = None
    test_y = None
    input_layer = None
    output_layer = None
    model = None

    def __init__(self,name):
        self.name = name
        self.model = NeuralModel(self.name)
    def setSamples(self,samples_x,samples_y):
        self.samples_x = samples_x
        self.samples_y = samples_y
        total_m = np.size(self.samples_x,0)
        train_m = int(0.7*total_m)
        cvd_m = int(0.9*total_m)
        self.setTrain(self.samples_x[:train_m,:],self.samples_y[:train_m,:])
        self.setCVD(self.samples_x[train_m:cvd_m,:],self.samples_y[train_m:cvd_m,:])
        self.setTest(self.samples_x[cvd_m:,:],self.samples_y[cvd_m:,:])

    def setTrain(self,train_x,train_y):
        '''
            Set training data, apply the training data to the model.
        :param train_x: Training input x
        :param train_y: Training output y
        :return: None
        '''
        self.train_x = train_x
        self.train_y = train_y
        self.model.SetTrainSamples(self.train_x,self.train_y)

    def setCVD(self,cvd_x,cvd_y):
        '''
            Set cross validation data, apply the cross validation data to the model's test data.
        :param cvd_x: Cross validation input x
        :param cvd_y: Cross validation output y
        :return: None
        '''
        self.cvd_x = cvd_x
        self.cvd_y = cvd_y
        self.model.SetTestSamples(self.cvd_x,self.cvd_y)
        pass

    def setTest(self,test_x,test_y):
        '''
            Set test data, for final test.
        :param test_x: Test input x
        :param test_y: Test output y
        :return: None
        '''
        self.test_x = test_x
        self.test_y = test_y

    def setFinalTest(self):
        '''
            Apply training and cross validation data to model's training data.
            Apply test data to model's test data.
        :return: None
        '''
        X = np.vstack((self.train_x,self.cvd_x))
        Y = np.vstack((self.train_y,self.cvd_y))
        self.model.SetTrainSamples(X,Y)
        self.model.SetTestSamples(self.test_x,self.test_y)

    def MiniBatch_Train(self,batch_size,steps,ifshow=True,gradiantCheck=False):
        '''
            A function for trainning in minibatch size
        :param batch_size:      Num of training data in one iteration
        :param steps:           Total training steps
        :param ifshow:          True if you want to print real-time info in console
        :param gradiantCheck:   False by default.
        :return:                True if the training meet no error.
        '''
        res = self.model.minibatch_train(batch_size,steps,ifshow,gradiantCheck)
        if res is False:
            print('Trainning Error!')
            return
        self.model.test_error()

    def Final_Train_and_Evaluate(self,batch_size,steps,ifshow=False):
        '''
            A function for final trainning.
            In this function, the system will use training and validation data to train your neural network model,
             and use test data to test error.
        :param batch_size:      Num of training data in one iteration
        :param steps:           Total training steps
        :param ifshow:          True if you want to print real-time info in console
        :return:                True if the training meet no error.
        '''
        self.setFinalTest()
        self.MiniBatch_Train(batch_size,steps,ifshow)














