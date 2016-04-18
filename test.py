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
    #y[y==0]=-1    #use tanh to classify
    yy = np.zeros([Na+Nb,2])
    for i in range(Na+Nb):
       if y[0,i]==1:
           yy[i,0] = 1
       else:
           yy[i,1] = 1
    myann.setSamples(x,yy)
    #myann.setSamples(x,y)
    myann.MiniBatch_Train(700,2000)

def DigitRecognitionTest():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_extract_mnist_data.load_dataset()
    # X_train = X_train.reshape(np.size(X_train,0),28*28).transpose()
    # X_val = X_val.reshape(np.size(X_val,0),28*28).transpose()
    # X_test = X_test.reshape(np.size(X_test,0),28*28).transpose()
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
    myann = NeuralNetwork('Digital Recognition')
    #myann.model.SetInputLayer(28*28)
    myann.model.SetInputLayer(1,28,28)
    myann.model.SetOutputLayer('SoftMax')
    #myann.model.SetConvolutionPoolingLayer([('ConvolutionLayer',(1,19,1)),('ConvolutionLayer',(1,6,1))])
    myann.model.SetConvolutionPoolingLayer([('ConvolutionLayer',(6,5,1)),('MaxPoolingLayer',(2,2)),
                                            ('ConvolutionLayer',(16,5,1)),('MaxPoolingLayer',(2,2)),
                                            ('ConvolutionLayer',(120,4,1))])
    myann.model.SetFullConnectLayer([(84,'ReLu'),(10,None)])
    #myann.model.SetFullConnectLayer([(28*28,'ReLu'),(30,'ReLu'),(20,'ReLu'),(10,None)])

    myann.setTrain(X_train,Y_train)
    myann.setCVD(X_val,Y_val)
    myann.setTest(X_test,Y_test)
    #myann.MiniBatch_Train(1000,20,ifshow=True,gradiantCheck=False)
    myann.MiniBatch_Train(30,2000,ifshow=True,gradiantCheck=False)

if __name__ == '__main__':
    DigitRecognitionTest()