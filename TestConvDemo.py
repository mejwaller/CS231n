import numpy as np
import unittest
import ConvDemo as conv;

#see convolution demo at http://cs231n.github.io/convolutional-networks/

class ConvDemoTest(unittest.TestCase):

    def setUp(self):
        self.W=5#image width
        self.H=5#image width
        self.D=3#image depth (rgb)
        self.F=3#convnet neuron field size
        self.S=2#convent neuron stride
        self.P=1#zero padding size
        self.K=2#number of filters
        self.IV=np.zeros((5,5,3))
        self.IV[:,:,0]=np.asarray([
                        [1,1,1,1,0],
                        [2,2,0,0,1],
                        [2,1,2,0,1],
                        [0,1,1,1,1],
                        [1,2,1,1,0]
                    ])
        self.IV[:,:,1]=np.asarray([
                        [1,2,0,0,2],
                        [2,0,2,1,2],
                        [1,1,0,0,0],
                        [1,0,1,2,2],
                        [2,0,2,0,1]
                    ])
        self.IV[:,:,2]=np.asarray([
                        [1,0,0,0,2],
                        [1,0,2,2,2],
                        [1,1,0,2,1],
                        [0,1,2,1,0],
                        [0,1,0,1,1]
                    ])

        self.W0=np.zeros((3,3,3))
        self.W0[:,:,0] = np.asarray([
                    [0,0,0],
                    [-1,0,0],
                    [-1,0,-1]
                ])
        self.W0[:,:,1]=np.asarray([
                    [1,0,1],
                    [-1,-1,1],
                    [-1,1,1]
                ])
        self.W0[:,:,2]=np.asarray([
                    [1,1,0],
                    [1,1,1],
                    [1,-1,1]
                ])
                
        self.W1=np.zeros((3,3,3))
        self.W1[:,:,0] = np.asarray([
                    [-1,-1,1],
                    [1,1,0],
                    [-1,0,-1]
                ])
        self.W1[:,:,1]=np.asarray([
                    [1,-1,-1],
                    [-1,-1,-1],
                    [-1,1,1],
                ])
        self.W1[:,:,0]=np.asarray([
                    [1,0,1],
                    [0,0,-1],
                    [1,1,-1]
                ])

        self.convnet= conv.net(self.W,self.H,self.D,self.F,self.S,self.P,self.K,self.IV)
                
    def testConstruction(self):
        cvnet= conv.net(self.W,self.H,self.D,self.F,self.S,self.P,self.K,self.IV)
        expW = self.W
        expH = self.H
        expD = self.D
        expF = self.F
        expS = self.S
        expP = self.P
        expK = self.K
        expIV = self.IV
        
        self.failUnless(cvnet.W == expW)
        self.failUnless(cvnet.H == expH)
        self.failUnless(cvnet.D == expD)
        self.failUnless(cvnet.F == expF)
        self.failUnless(cvnet.S == expS)
        self.failUnless(cvnet.P == expP)
        self.failUnless(cvnet.K == expK)
        self.failUnless(np.array_equal(cvnet.IV,expIV))
        
    def testSetWrongFilter(self):
        with self.assertRaises(conv.FilterError):
            self.convnet.setFilter(self.W0,3)
            
    def testSetFilter(self):        
        self.convnet.setFilter(self.W0,0)
        self.convnet.setFilter(self.W1,1)
        self.failUnless(np.array_equal(self.W0,self.convnet.W0))
        self.failUnless(np.array_equal(self.W1,self.convnet.W1))
        
    def testPadding(self):
        self.failUnless(self.convnet.padded.shape[0]==7)
        self.failUnless(self.convnet.padded.shape[1]==7)
        self.failUnless(self.convnet.padded.shape[2]==3)
        
    def testPaddedData(self):
        self.failUnless(np.array_equal(self.convnet.IV,self.convnet.padded[self.convnet.P:self.convnet.padded.shape[0]-self.convnet.P,self.convnet.P:self.convnet.padded.shape[1]-self.convnet.P,:]))
        
        
        
        
        
        
           
if __name__ == '__main__': unittest.main()       
            
            
            
                    
                        
                        
        
    
    