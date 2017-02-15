import gradient as g
import numpy as np

W = np.random.rand(3073, 10) * 0.001 # random weight vector

df = g.numerical_grad(g.svm_loss_fun,W)

loss_original = g.svm_loss_fun(W) # the original loss
print 'original loss: %f' % (loss_original, )

# lets see the effect of multiple step sizes
for step_size_log in [-10, -9, -8, -7, -6, -5,-4,-3,-2,-1]:
  step_size = 10 ** step_size_log
  W_new = W - step_size * df # new position in the weight space
  loss_new = g.svm_loss_fun(W_new)
  print 'for step size %f new loss: %f' % (step_size, loss_new)