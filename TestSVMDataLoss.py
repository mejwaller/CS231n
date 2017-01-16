import unittest, SVMDataLoss
import numpy as np
from cifar10_preproc import PreProcCifar10 as preproc 

class myTest(unittest.TestCase):
    
    #see http://cs231n.github.io/linear-classify/ 
    #(section Multiclass Support Vector Machine loss)  
    def setUp(self):
        self.pp=preproc()
        self.pp.preProcess()
        xs=[13,-7,11,1]
        self.xtest=np.array(xs)
        #Ws=[[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        #self.Wtest = np.array(Ws)
        self.Wtest = np.eye(3,4)
        self.thing=6
        self.Wpreproc = np.eye(10,3073)*2
    
    #check our test Wx is correct   
    def testWdotx(self):
        expected=np.array([13,-7,11])
        res = self.Wtest.dot(self.xtest)
        self.failUnless(np.array_equal(expected,res))    
        
    #check loss for this example is 8    
    def testL_i(self): 
        expected = 8
        self.failUnless(SVMDataLoss.L_i(self.xtest,0,self.Wtest,10)==8)  
        
    def testunvectorizedL_ifortestset(self):#get loss for first image of preprocessor fixed dev set
        print SVMDataLoss.L_i(self.pp.fixedX_dev[1],self.pp.fixedy_dev[1], self.Wpreproc)   
        
    def testunvectorisedL_iforFullTestSet(self):# get total loss               
        score = 0;
        for i in xrange(500):
            score+=SVMDataLoss.L_i(self.pp.fixedX_dev[i], self.pp.fixedy_dev[i],self.Wpreproc)
        print score
    
        
if __name__ == '__main__': unittest.main()