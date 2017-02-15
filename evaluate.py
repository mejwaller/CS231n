import gradient as g
import numpy as np

W = np.random.rand(3073, 10) * 0.001 # random weight vector

df = g.numerical_grad(g.svm_loss_fun,W)