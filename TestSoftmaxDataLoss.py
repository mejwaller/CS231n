import unittest, SoftmaxDataLoss, SVMDataLoss
import numpy as np
from cifar10_preproc import PreProcCifar10 as preproc 
import time

class SoftmaxTest(unittest.TestCase):
    
    #see http://cs231n.github.io/linear-classify/ 
    #(section Multiclass Support Vector Machine loss)  
    def setUp(self):    
        self.pp=preproc()
        self.pp.preProcess()
        self.pp.addBias()
        #setup toy W and x (augmented with b) - data from link above        
        self.xtoy=np.array([-15,22,-44,56,1])
        self.Wtoy = np.array([[0.01,-0.05,0.1,0.05,0.0],[0.7,0.2,0.05,0.16,0.2],[0.0,-0.45,-0.2,0.03,-0.3]]).transpose()
        self.Wpreproc = np.eye(3073,10)*2      
        
    def testLi_unvec(self):
        score = SoftmaxDataLoss.Li_unvec(self.xtoy,2,self.Wtoy)
        print "Score is %f" % score
        self.failUnless(np.isclose(score,1.0401905694301092))
        
    def testLi_unvecFulldevset(self):
        score = 0;
        tic = time.time()
        for i in xrange(500):
            score+=SoftmaxDataLoss.Li_unvec(self.pp.fixedX_dev[i], self.pp.fixedy_dev[i],self.Wpreproc)
        score/=500 
        score+=SVMDataLoss.regL2norm(self.Wpreproc,1.)   
        toc = time.time()
        print "UNVECTORIZED Score is %f" % score
        print "Unvectorized softmax took %fs" % (toc -tic)        
        self.failUnless(np.isclose(score,74.196696))
        
    def testLVec(self):
        score = 0;
        tic = time.time()
        score,probs=SoftmaxDataLoss.SoftmaxLoss(self.pp.fixedX_dev, self.pp.fixedy_dev,self.Wpreproc)
        #score/=500 
        #score+=SVMDataLoss.regL2norm(self.Wpreproc,1.)   
        toc = time.time()
        print "VECTORIZED Score is %f" % score
        print "Vectorized softmax took %fs" % (toc -tic)        
        self.failUnless(np.isclose(score,74.196696))
        
        
if __name__ == '__main__': unittest.main()
        
        
    
    
    