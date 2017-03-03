import numpy as np
import matplotlib.pyplot as plt

class spiralDataSet(object):
    def __init__(self):
        pass
        
    def generateData(self,n=100,k=3, d=2):#n=no. of points per class, k=n. of classes, d = dimenisonality
        self.X = np.zeros((n*k, d))#data matrix = each row is a single example
        self.y = np.zeros(n*k,dtype='uint8')
        for j in xrange(k):
            ix = range(n*j,n*(j+1))#indexes into X for class j
            r = np.linspace(0.0,1,n)#radius - how far from the center (0.0 is center, 1 is max radius)
            t = np.linspace(j*4,(j+1)*4,n) + np.random.randn(n)*0.2#angle (theta) + a random perturbation
            self.X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
            self.y[ix]=j
            
    def plot(self):
        plt.scatter(self.X[:,0],self.X[:,1], c= self.y, s=40, cmap=plt.cm.Spectral)
        plt.show()
        
                    
            
            
        
        
        