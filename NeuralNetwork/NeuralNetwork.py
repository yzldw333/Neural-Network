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
        self.model = NeuralModel()
    def setSamples(self,samples_x,samples_y):
        self.samples_x = samples_x
        self.samples_y = samples_y
        total_m = np.size(self.samples_x,1)
        train_m = int(0.7*total_m)
        cvd_m = int(0.9*total_m)
        self.setTrain(self.samples_x[:, :train_m],self.samples_y[:, :train_m])
        self.setCVD(self.samples_x[:,  train_m:cvd_m],self.samples_y[:,  train_m:cvd_m])
        self.setTest(self.samples_x[:, cvd_m:],self.samples_y[:, cvd_m:])

    def setTrain(self,train_x,train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.model.SetTrainSamples(self.train_x,self.train_y)

    def setCVD(self,cvd_x,cvd_y):
        self.cvd_x = cvd_x
        self.cvd_y = cvd_y
        self.model.SetTestSamples(self.cvd_x,self.cvd_y)
        pass

    def setTest(self,test_x,test_y):
        self.test_x = test_x
        self.test_y = test_y

    def setFinalTest(self):
        X = np.hstack(self.train_x,self.cvd_x)
        Y = np.hstack(self.train_y,self.cvd_y)
        self.model.SetTrainSamples(X,Y)
        self.model.SetTestSamples(self.test_x,self.test_y)

    def MiniBatch_Train(self,batch_size,steps,ifshow=False):
        res = self.model.minibatch_train(batch_size,steps,ifshow)
        if res is False:
            return
        self.model.test_error()

    def Final_Train_and_Evaluate(self,batch_size,steps,ifshow=False):
        self.setFinalTest()
        self.MiniBatch_Train(batch_size,steps,ifshow)














