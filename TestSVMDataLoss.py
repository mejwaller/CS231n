import unittest, SVMDataLoss
import numpy as np
from cifar10_preproc import PreProcCifar10 as preproc 

class myTest(unittest.TestCase):
    
    #see http://cs231n.github.io/linear-classify/ 
    #(section Multiclass Support Vector Machine loss)  
    def setUp(self):
        pp=preproc()
        pp.preProcess()
        xs=[13,-7,11,1]
        self.xtest=np.array(xs)
        #Ws=[[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        #self.Wtest = np.array(Ws)
        self.Wtest = np.eye(3,4)
        self.thing=6
    
    #check our test Wx is correct   
    def testWdotx(self):
        expected=np.array([13,-7,11])
        res = self.Wtest.dot(self.xtest)
        self.failUnless(np.array_equal(expected,res))    
        
    #check loss for this example is 8    
    def testL_i(self): 
        expected = 8
        self.failUnless(SVMDataLoss.L_i(self.xtest,0,self.Wtest,10)==8)  
        
    
        
if __name__ == '__main__': unittest.main()