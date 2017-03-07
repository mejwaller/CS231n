import numpy as np
import matplotlib.pyplot as plt
import SpiralDataSet

k= 3
d = 2
h=100#size of hidden layer

sp = SpiralDataSet.spiralDataSet()
sp.generateData(h,k,d)

X = sp.X

#X=np.hstack((Xraw,np.ones((Xraw.shape[0],1))))#append bias
y = sp.y

W=0.01*np.random.randn(d,h)
b=np.zeros((1,h))
W2 = 0.01*np.random.randn(h,k)
b2=np.zeros((1,k))

step_size = 1
reg=1e-3

num_examples=X.shape[0]
#gradient descent loop
for i in xrange(10000):
    
    #evaluate class scores
    hidden_layer = np.maximum(0, np.dot(X,W) + b)
    scores=np.dot(hidden_layer,W2) + b2
    
    #compute class probabilities
    exp_scores=np.exp(scores)
    probs=exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    
    #compute the loss
    correct_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss+reg_loss
    if i%1000 == 0:
        print "iteration %d: loss %f" % (i,loss)
        
    #compute graduient on scores
    dscores=probs
    dscores[range(num_examples),y]-=1
    dscores/=num_examples
    
    
    #backprop the grads to the paramnters
    #start with W2 and b2:
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    #hidden layer:
    dhidden = np.dot(dscores, W2.T)
    #backprop ReLU non-linearity
    dhidden[hidden_layer<=0]=0
    #W and b:
    dW=np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0,keepdims=True)
    
    #add reg
    dW2+=reg*W2
    dW+=reg*W
    
    #update params
    W += -step_size*dW
    b += -step_size*db
    W2 += -step_size*dW2
    b2 += -step_size*db2
    
# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
#fig.savefig('spiral_net.png')
    
    
    
    
    
    
    