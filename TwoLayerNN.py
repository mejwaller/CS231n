#heavily plagiarised from https://github.com/bruceoutdoors/CS231n/blob/master/assignment1/cs231n/classifiers/neural_net.py

import numpy as np

#two-layer (one hidden layer) nn with ReLU activation and softmax loss
class TwoLayerNN(object):
    def __init__(self, inputsize, hiddensize, numclasses, perturb=1e-2):
        """
        inputsize - dimensionality D of the data
        hiddensize - number of neurons (H) in teh hidden layer
        numclasses - number of classes (C) to classify
        perturb - small nunber to ensure initial weights are not all zero but are small and random
        """
        
        #first layer weights - shape D x H 
        #use small random number for intialization        
        self.W1=perturb*np.random.randn(inputsize,hiddensize)
        #calibrate using 1/sqrt(n)
        #self.W1=np.random.randn(inputsize,hiddensize)/sqrt(inputsize)

        #For ReLU neurions: calibrate using sqrt(2/n)
        #self.W1=np.random.randn(inputsize,hiddensize)*sqrt(2.0/inputsize)        
        
        #first layer bias vector
        self.b1 = np.zeros(hiddensize)
        
        #hidden layer weights - shape H x C  
        #use small random number for intialization        
        self.W2=perturb*np.random.randn(hiddensize,numclasses)
        #calibrate using 1/sqrt(n)
        #self.W2=np.random.randn(hiddensize,numclasses)/sqrt(hiddensize)  
        
        #hidden layer bias vector
        self.b2=np.zeros(numclasses)
        
    def loss(self,X, y=None, reg=0.1):
        """
        X = input data N x D. Each X[i] is training sample (N of them)
        y = vector of training labels. Each y[i] is an integer in range 0 <= y[i] < C
        If y is not passed, just retunr scores, else return loss and gradients
        """
        
        N, D = X.shape
        
        #forward pass:
        z1=X.dot(self.W1) + self.b1
        a1=np.maximum(0,z1)#ReLU activation
        scores=a1.dot(self.W2)+self.b2
        
        #if targets not given, just return scores
        if y is None:
            return scores
            
        #compute the loss
        exp_scores=np.exp(scores)
        probs=exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(N),y])
        data_loss = np.sum(correct_logprobs)/N
        reg_loss = 0.5*(reg*np.sum(self.W1*self.W1)+reg*np.sum(self.W2*self.W2))
        
        loss = data_loss + reg_loss
        
        #back pass - compute the gradients
        grads={}
        
        return loss, grads
        
        
        
    
        


