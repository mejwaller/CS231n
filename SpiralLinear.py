import SpiralDataSet
import SoftmaxDataLoss as sm
import numpy as np
import matplotlib.pyplot as plt

n= 100
k= 3
d = 2

sp = SpiralDataSet.spiralDataSet()
sp.generateData(n,k,d)

X = sp.X

#X=np.hstack((Xraw,np.ones((Xraw.shape[0],1))))#append bias
y = sp.y

W = 0.01*np.random.randn(d,k)
b= np.zeros((1,k))

#W = np.vstack((Wraw,b))#append bias


print "W shape is: ",W.shape
print "X shape is: ",X.shape


#hyperparameters
step_size = 1;
reg = 1e-3#regulariztaion strength

num_examples=X.shape[0]

for i in xrange(400):

    loss, probs = sm.SoftmaxLossBias(X,y,W,b,reg)

    if i%d == 0:
        print "Iteration: %d: Loss: %f" % (i,loss)
        
    #compute gradients
    dscores=probs
    dscores[range(num_examples),y]-=1
    dscores/=num_examples
    
    #backprop the grdaient ot parameters W and b
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    
    dW+=reg*W
    
    W+= -step_size*dW
    b += -step_size*db
    

scores = np.dot(X,W) + b
predicted_class = np.argmax(scores, axis=1)
print "Training accuracy: %.2f" % (np.mean(predicted_class==y))


# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
#fig.savefig('spiral_linear.png')



