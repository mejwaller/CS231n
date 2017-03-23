#heavily plagiarised from https://github.com/bruceoutdoors/CS231n/blob/master/assignment1/cs231n/classifiers/neural_net.py

import numpy as np

#two-layer (one hidden layer) nn with ReLU activation and softmax loss
class TwoLayerNN(object):
    def __init__(self, inputsize, hiddensize, numclasses, perturb=1e-2):
        """
        inputsize - dimensionality D of the data
        hiddensize - number of neurons (H) in teh hidden layer
        numclasses - number of classes (C) to classify
        perturb - small nunber to ensure initial weights are not all zero but are small and random
        """
        
        self.params = {}
        
        #first layer weights - shape D x H 
        #use small random number for intialization        
        self.params['W1']=perturb*np.random.randn(inputsize,hiddensize)
        #calibrate using 1/sqrt(n)
        #self.params['W1']=np.random.randn(inputsize,hiddensize)/sqrt(inputsize)

        #For ReLU neurions: calibrate using sqrt(2/n)
        #self.params['W1']=np.random.randn(inputsize,hiddensize)*sqrt(2.0/inputsize)        
        
        #first layer bias vector
        self.params['b1'] = np.zeros(hiddensize)
        
        #hidden layer weights - shape H x C  
        #use small random number for intialization        
        self.params['W2']=perturb*np.random.randn(hiddensize,numclasses)
        #calibrate using 1/sqrt(n)
        #self.params['W2']=np.random.randn(hiddensize,numclasses)/sqrt(hiddensize)  
        
        #hidden layer bias vector
        self.params['b2']=np.zeros(numclasses)
        
    def loss(self,X, y=None, reg=0.1):
        """
        X = input data N x D. Each X[i] is training sample (N of them)
        y = vector of training labels. Each y[i] is an integer in range 0 <= y[i] < C
        If y is not passed, just retunr scores, else return loss and gradients
        """
        
        N, D = X.shape
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        #forward pass:
        z1=X.dot(W1) + b1
        a1=np.maximum(0,z1)#ReLU activation
        scores=a1.dot(W2)+b2
        
        #subtract max value for the scores for each image
        #scores-=np.max(scores,axis=1)[:, np.newaxis]#subtract max value for the scores for each image
        
        #
        
        #if targets not given, just return scores
        if y is None:
            return scores
            
        #compute the loss
        exp_scores=np.exp(scores)
        probs=exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(N),y])
        data_loss = np.sum(correct_logprobs)/N
        reg_loss = 0.5*(reg*np.sum(W1*W1)+reg*np.sum(W2*W2))
        
        loss = data_loss + reg_loss
        
        #back pass - compute the gradients
        grads={}
        
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # compute the gradient on scores
        dscores = probs
        dscores[range(N),y] -= 1
        dscores /= N

        # W2 and b2
        grads['W2'] = np.dot(a1.T, dscores)
        grads['b2'] = np.sum(dscores, axis=0)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[a1 <= 0] = 0
        # finally into W,b
        grads['W1'] = np.dot(X.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1
        
        return loss, grads
        
    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
        X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
        after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
    
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None
    
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
    
            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
    
            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
    
                # Decay learning rate
                learning_rate *= learning_rate_decay
    
        return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
        classify.
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
        the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
        to have class c, where 0 <= c < C.
        """
        y_pred = None
    
        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1) # pass through ReLU activation function
        scores = a1.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################
    
        return y_pred
            
        
    
        


