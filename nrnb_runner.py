import data_utils as du
import NearestNeighbour as nb
import numpy as np

Xtr, Ytr, Xte, Yte = du.load_CIFAR10('datasets/cifar-10-batches-py/')

Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3) #Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3) # Xte_rows becomes 10000 x 3072

nn = nb.NearestNeighbour()

nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels

Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# pf examples that are correctly predicted (i.e. label matches)

print 'accuracy: %f' % (np.mean(Yte_predict == Yte))
