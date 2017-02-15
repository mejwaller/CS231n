import numpy as np
from cifar10_preproc import PreProcCifar10 as preproc
import SVMDataLoss, SoftmaxDataLoss  

def numerical_grad(f, x, h=0.00001):
    """ 
    - a naive implementation of numerical gradient of f at x 
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    grad = np.zeros(x.shape)
  
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # increment by h
        fxplush = f(x) # evalute f(x + h)
    
        # evaluate funciton at x-h
        x[ix] = old_value - h
        fxminush = f(x)
        x[ix] = old_value # restore to previous value (very important!)
    
        # compute the partial derivative
        grad[ix] = (fxplush - fxminush) / (2*h) # the slope
        it.iternext() # step to next dimension

    return grad
    
pp = preproc()
pp.preProcess()

def svm_loss_fun(W):
    SVMDataLoss.SVM_loss(pp.X_train,pp.y_train,W)
    




    



