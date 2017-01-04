import unittest, SVMDataLoss
import numpy as np

class myTest(unittest.TestCase):
    
    def setUp(self):
        xs=[13,-7,11,1]
        self.xtest=np.array(xs)
        Ws=[[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        self.Wtest = np.array(Ws)
        self.thing=6
        
    def testWdotx(self):
        expected=np.array([13,-7,11])
        res = self.Wtest.dot(self.xtest)
        print expected, res
        self.failUnless(np.array_equal(expected,res))       
        
if __name__ == '__main__': unittest.main()