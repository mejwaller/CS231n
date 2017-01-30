#see https://github.com/bruceoutdoors/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py

import numpy as np

def L_i(x, y, W, delta=1):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 1 x 3073 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 3073 x 10)
  """
  #NB: altered delta from 1 to 10 for testing
  #delta = 10.0 # see notes about delta later in this section
  scores = x.dot(W) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[1] # number of classes, e.g. 10
  loss_i = 0.0
  for j in xrange(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i
  
def L_i_vectorized(x, y, W, delta = 1):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    scores = x.dot(W)
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

def SVM_loss(X,y,W, delta=1):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 50,000  x 3073 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 3073 x 10)
    """
    scores = X.dot(W)
    correct_class_score = scores[np.arange(X.shape[0]),y]
    margins = np.maximum(0,scores - correct_class_score[:, np.newaxis] + delta)
    margins[np.arange(X.shape[0]),y]=0
    loss = np.sum(margins)
    loss/=X.shape[0]
    loss += regL2norm(W,1.)
    return loss    
    
def regL2norm(W,reg):
    """ 
    sum up the squared elements of W
    multiply by reg, the regularization constant
    """
    return reg*np.sum(W*W)

