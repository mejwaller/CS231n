import numpy as np
from cifar10_preproc import PreProcCifar10 as preproc
import SVMDataLoss, SoftmaxDataLoss  

pp = preproc()

pp.preProcess()

bestloss = float("inf")

W = np.random.randn(3073, 10) * 0.0001 # generate random parameters

for num in xrange(1000):
  step=0.0001
  Wtry = W + np.random.randn(3073, 10) * step
  #W = np.random.randn(3073, 10) * 0.0001 # generate random parameters
  loss = SVMDataLoss.SVM_loss(pp.X_train, pp.y_train, Wtry) # get the loss over the entire training set
  #loss = SVMDataLoss.SVM_loss(pp.X_dev, pp.y_dev, Wtry) # get the loss over the entire training set
  #loss = SoftmaxDataLoss.SoftmaxLoss(pp.X_dev, pp.y_dev, Wtry) # get the loss over the entire training set
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    #bestW = W
    W = Wtry
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

scores=pp.X_test.dot(Wtry)

print scores

Yte_predict = np.argmax(scores, axis = 1)

#print Yte_predict

#print pp.y_test

#print Yte_predict.shape
#print pp.y_test.shape

print 'accuracy: %f' % (np.mean(Yte_predict == pp.y_test))
