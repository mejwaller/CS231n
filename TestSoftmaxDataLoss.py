import unittest, SVMDataLoss
import numpy as np
from cifar10_preproc import PreProcCifar10 as preproc 
import time

class SoftmaxTest(unittest.TestCase):
    
    #see http://cs231n.github.io/linear-classify/ 
    #(section Multiclass Support Vector Machine loss)  
    def setUp(self):    
        self.pp=preproc()
        self.pp.preProcess()
        #setup toy W and x (augmented with b) - data from link above        
        self.xtoy=np.array([-15,22,-44,56,1])
        Wtoy = np.array([[0.01,-0.05,0.1,0.05,0.0],[0.7,0.2,0.05,0.16,0.2],[0.0,-0.45,-0.2,0.03,-0.3]]).transpose()
        self.Wpreproc = np.eye(3073,10)*2
    
    