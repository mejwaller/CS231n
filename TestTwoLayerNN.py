import numpy as np
import matplotlib.pyplot as plt

from TwoLayerNN import TwoLayerNN

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x,y):
    return np.max(np.abs(x-y)/(np.maximum(1e-8, np.abs(x) + np.abs(y))))


#create a small net and some toy data to checl implementation

inputsize=4
hiddensize=10
numclasses=3
numinputs=5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNN(inputsize,hiddensize,numclasses,1e-1)
    
def init_toy_data():
    np.random.seed(1)
    X=10*np.random.randn(numinputs,inputsize)
    y=np.array([0,1,2,2,1])
    return X,y
    
net = init_toy_model()
X, y = init_toy_data()

#forward pass - check scores
scores = net.loss(X)
print 'Your scores:'
print scores
print
print 'correct scores:'
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print correct_scores
print

# The difference should be very small. We get < 1e-7
print 'Difference between your scores and correct scores:'
print np.sum(np.abs(scores - correct_scores))
    
#check loss
loss, _ = net.loss(X, y, reg=0.1)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print 'Difference between your loss and correct loss:'
print np.sum(np.abs(loss - correct_loss))