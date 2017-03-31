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
        self.IV = np.asarray([
                    [
                        [1,1,1,1,0],
                        [2,2,0,0,1],
                        [2,1,2,0,1],
                        [0,1,1,1,1],
                        [1,2,1,1,0]
                    ],
                    [
                        [1,2,0,0,2],
                        [2,0,2,1,2],
                        [1,1,0,0,0],
                        [1,0,1,2,2],
                        [2,0,2,0,1]
                    ],
                    [
                        [1,0,0,0,2],
                        [1,0,2,2,2],
                        [1,1,0,2,1],
                        [0,1,2,1,0],
                        [0,1,0,1,1]
                    ]
                ])#convdemo input
                
        self.W0 = np.asarray([
                [
                    [0,0,0],
                    [-1,0,0],
                    [-1,0,-1]
                ],
                [
                    [1,0,1],
                    [-1,-1,1],
                    [-1,1,1]
                ],
                [
                    [1,1,0],
                    [1,1,1],
                    [1,-1,1]
                ]
            ])
        self.W1 = np.asarray([
                [
                    [-1,-1,1],
                    [1,1,0],
                    [-1,0,-1]
                ],
                [
                    [1,-1,-1],
                    [-1,-1,-1],
                    [-1,1,1],
                ],
                [
                    [1,0,1],
                    [0,0,-1],
                    [1,1,-1]
                ]
            ])
                
    def testConstruction(self):
        convnet= conv.net(self.W,self.H,self.D,self.F,self.S,self.P,self.K,self.IV)
        expW = self.W
        expH = self.H
        expD = self.D
        expF = self.F
        expS = self.S
        expP = self.P
        expK = self.K
        expIV = self.IV
        
        self.failUnless(convnet.W == expW)
        self.failUnless(convnet.H == expH)
        self.failUnless(convnet.D == expD)
        self.failUnless(convnet.F == expF)
        self.failUnless(convnet.S == expS)
        self.failUnless(convnet.P == expP)
        self.failUnless(convnet.K == expK)
        self.failUnless(np.array_equal(convnet.IV,expIV))
        
    def testSetWrongFilter(self):
        convnet= conv.net(self.W,self.H,self.D,self.F,self.S,self.P,self.K,self.IV)
        with self.assertRaises(conv.FilterError):
            convnet.setFilter(self.W0,3)
        
        
        
           
if __name__ == '__main__': unittest.main()       
            
            
            
                    
                        
                        
        
    
    