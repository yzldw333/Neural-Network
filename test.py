from NeuralNetwork import *
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.Layers.core import *
from initialize import *
import load_and_extract_mnist_data


def DoubleMoonTest():
    myann = NeuralNetwork('Dong Dong Network')
    myann.model.SetInputLayer(2)
    myann.model.SetOutputLayer('SoftMax')
    myann.model.SetFullConnectLayer([(15,'ReLu'),(2,None)])
    Na = 500
    Nb = 500
    (x,y) = initialize(10,6,-4,Na,Nb)
    x = x.transpose()
    yy = np.zeros([Na+Nb,2])
    for i in range(Na+Nb):
       if y[0,i]==1:
           yy[i,0] = 1
       else:
           yy[i,1] = 1
    myann.setSamples(x,yy)
    myann.MiniBatch_Train(700,2000)

def LeNet5():
    myann = NeuralNetwork('LeNet5')
    myann.model.SetInputLayer(1,28,28)
    myann.model.SetOutputLayer('SoftMax')
    myann.model.SetConvolutionPoolingLayer([('ConvolutionLayer',(6,5,1)),('MaxPoolingLayer',(2,2)),
                                            ('ConvolutionLayer',(16,5,1)),('MaxPoolingLayer',(2,2)),
                                            ('ConvolutionLayer',(120,4,1))])
    myann.model.SetFullConnectLayer([(84,'ReLu'),(10,None)])
    return myann
def Fake_LeNet5():
    myann = NeuralNetwork('FakeLeNet5')
    myann.model.SetInputLayer(1,28,28)
    myann.model.SetOutputLayer('SoftMax')
    myann.model.SetConvolutionPoolingLayer([('ConvolutionLayer',(120,5,1)),('MaxPoolingLayer',(2,2)),
                                            ('ConvolutionLayer',(120,5,1)),('MaxPoolingLayer',(2,2)),
                                            ('ConvolutionLayer',(120,4,1))])
    myann.model.SetFullConnectLayer([(84,'ReLu'),(10,None)])
    return myann

def TestNet():
    myann = NeuralNetwork('TestNet')
    myann.model.SetInputLayer(1,28,28)
    myann.model.SetOutputLayer('SoftMax')
    myann.model.SetConvolutionPoolingLayer([('ConvolutionLayer',(20,5,1)),('MaxPoolingLayer',(2,2)),
                                            ('ConvolutionLayer',(50,5,1)),('MaxPoolingLayer',(2,2))])
    myann.model.SetFullConnectLayer([(500,'ReLu'),(10,None)])
    return myann

def DigitRecognitionTest():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_extract_mnist_data.load_dataset()
    Y_train = []
    for e in y_train:
        tmp = [0]*10
        tmp[e]=1
        Y_train.append(tmp)
    Y_train = np.array(Y_train)
    Y_val = []
    for e in y_val:
        tmp = [0]*10
        tmp[e]=1
        Y_val.append(tmp)
    Y_val = np.array(Y_val)
    Y_test = []
    for e in y_test:
        tmp = [0]*10
        tmp[e]=1
        Y_test.append(tmp)
    Y_test = np.array(Y_test)
    myann = LeNet5()
    myann.setTrain(X_train,Y_train)
    myann.setCVD(X_val,Y_val)
    myann.setTest(X_test,Y_test)
    #myann.MiniBatch_Train(1000,30,ifshow=True,gradiantCheck=False) #use cross validation data to test
    myann.Final_Train_and_Evaluate(1000,30,ifshow=True) #use test data to test

if __name__ == '__main__':
    #DoubleMoonTest()
    DigitRecognitionTest()