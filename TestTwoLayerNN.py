import numpy as np
import matplotlib.pyplot as plt
import unittest

from TwoLayerNN import TwoLayerNN

from gradient_check import eval_numerical_gradient

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class TwoLayerNNTest(unittest.TestCase):
    
    def setUp(self):
        #create a small net and some toy data to check implementation
        self.inputsize=4
        self.hiddensize=10
        self.numclasses=3
        self.numinputs=5
        np.random.seed(0)
        self.nn2lyr=TwoLayerNN(self.inputsize,self.hiddensize,self.numclasses,1e-1)
        np.random.seed(1)
        self.X=10*np.random.randn(self.numinputs, self.inputsize)
        self.y = np.array([0,1,2,2,1])
        
    def testScores(self):
        correct_scores = np.asarray([
            [-0.81233741, -1.27654624, -0.70335995],            
            [-0.17129677, -1.18803311, -0.47310444],
            [-0.51590475, -1.01354314, -0.8504215 ],
            [-0.15419291, -0.48629638, -0.52901952],
            [-0.00618733, -0.12435261, -0.15226949]])
        scores=self.nn2lyr.loss(self.X)
        self.failUnless(np.allclose(scores,correct_scores))
        
    def testLoss(self):
        correct_loss = 1.30378789133
        loss, _ = self.nn2lyr.loss(self.X, self.y, reg=0.1)
        self.failUnless(np.isclose(loss,correct_loss))
        
    def testGrads(self):
        loss, grads = self.nn2lyr.loss(self.X, self.y, reg=0.1)
        # these should all be less than 1e-8 or so
        for param_name in grads:
            f = lambda W: self.nn2lyr.loss(self.X, self.y, reg=0.1)[0]
            param_grad_num = eval_numerical_gradient(f, self.nn2lyr.params[param_name], verbose=False)
            self.failUnless(self.rel_error(param_grad_num, grads[param_name]) < 1e-8)
            #print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
        
    
    def rel_error(self,x,y):
        return np.max(np.abs(x-y)/(np.maximum(1e-8, np.abs(x) + np.abs(y))))
        
if __name__ == '__main__': unittest.main()


"""
#train toy model
net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=1e-5,
            num_iters=100, verbose=False)

print 'Final training loss: ', stats['loss_history'][-1]

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()
"""