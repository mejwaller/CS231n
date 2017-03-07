import unittest, SVMDataLoss
import numpy as np
from cifar10_preproc import PreProcCifar10 as preproc 
import time

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
        self.Wtest = np.eye(4,3)
        self.thing=6
        self.Wpreproc = np.eye(3073,10)*2
    
    #check our test Wx is correct   
    def testWdotx(self):
        expected=np.array([13,-7,11])
        res = self.xtest.dot(self.Wtest)
        self.failUnless(np.array_equal(expected,res))    
        
    #check loss for this example is 8    
    def testL_i(self): 
        expected = 8
        self.failUnless(SVMDataLoss.L_i(self.xtest,0,self.Wtest,10)==expected) 
        
         #check loss for this example is 8    
    def testL_i_vectorized(self): 
        expected = 8
        self.failUnless(SVMDataLoss.L_i_vectorized(self.xtest,0,self.Wtest,10)==expected) 
        
    def testunvectorizedL_ifortestset(self):#get loss for first image of preprocessor fixed dev set
        loss=SVMDataLoss.L_i(self.pp.fixedX_dev[1],self.pp.fixedy_dev[1], self.Wpreproc)            
        self.failUnless(np.isclose(loss, 144.886612245))
 
    def testVectorizedL_ifortestset(self):#get loss for first image of preprocessor fixed dev set
           self.failUnless(np.isclose(SVMDataLoss.L_i_vectorized(self.pp.fixedX_dev[1],self.pp.fixedy_dev[1], self.Wpreproc), 144.886612245))
       
    def testunvectorisedL_iforFullTestSet(self):# get total loss                       
        score = 0;
        tic = time.time()
        for i in xrange(500):
            score+=SVMDataLoss.L_i(self.pp.fixedX_dev[i], self.pp.fixedy_dev[i],self.Wpreproc)
        score/=500 
        score+=SVMDataLoss.regL2norm(self.Wpreproc,1.)   
        toc = time.time()
        print "Unvectorized took %fs" % (toc -tic)       
        #self.failUnless(np.isclose(score,108097.158163))
        self.failUnless(np.isclose(score,236.194316327))
  
        
    def testVectorisedL_iforFullTestSet(self):# get total loss               
        score = 0;
        tic = time.time()
        for i in xrange(500):
            score+=SVMDataLoss.L_i_vectorized(self.pp.fixedX_dev[i], self.pp.fixedy_dev[i],self.Wpreproc)
        score/=500 
        score+=SVMDataLoss.regL2norm(self.Wpreproc,1.) 
        toc = time.time()
        print "Semi-vectorized took %fs" % (toc -tic)
        #self.failUnless(np.isclose(score,108097.158163))
        self.failUnless(np.isclose(score,236.194316327))
        
    def testFullvectorizedLforFullTestSet(self):
        score = 0;
        tic=time.time()
        score=SVMDataLoss.SVM_loss(self.pp.fixedX_dev,self.pp.fixedy_dev,self.Wpreproc)
        toc=time.time()
        print "fully vectorized took %fs" % (toc-tic)
        self.failUnless(np.isclose(score,236.194316327))
       
    
        
if __name__ == '__main__': unittest.main()