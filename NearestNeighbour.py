import numpy as np

class NearestNeighbour(object):
    def __init__(self):
        pass
        
    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimensioanl of size N "(the labels in numeric form)"""
        # the nearest neighbour classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X):
        """X is N x D where each row is an example we wish to predict label for"""
        num_test=X.shape[0]
        # Let's make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        
        for i in xrange(num_test):
            # find the nearest training image to the 'i'th test image
            # using the L1 distance (sum of abolute value differences)
            distances = np.sum(np.abs(self.Xtr-X[i,:]), axis =1)
            min_index = np.argmin(distances) # get the index with the smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            print 'Done row %d' % (i)
            
        return Ypred