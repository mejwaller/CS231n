import numpy as np
import SVMDataLoss

def Li_unvec(x,y,W):
    scores = x.dot(W)
    #see http://cs231n.github.io/linear-classify/#softmax, 'Practical issues: Numeric stability' 
    scores-=np.max(scores)
    p = np.exp(scores)/np.sum(np.exp(scores))
    loss_i = np.log(p[y]) #p[y] are the 'correct' class scores
    return loss_i
    
def SoftmaxLoss(X,y,W):
    """
    X = data - e.g. 500 x 3073 (500 images, 3072 pixel values per image + b vec appended)
    y = 500 - correct class for each image
    W = 3073 x 10 - weight matrix. 10 values for each class for each pixel
    """
    scores = X.dot(W)#500x10 - scores for each image for each class
    scores-=np.max(scores,axis=1)[:, np.newaxis]#subtract max value for the scores for each image
    
    p = np.exp(scores)/np.sum(np.exp(scores),axis=1)[:,np.newaxis]
    
    correct_class_score = p[np.arange(p.shape[0]),y]
    
    loss=sum(np.log(correct_class_score))/p.shape[0]
    
    loss+=SVMDataLoss.regL2norm(W,1.)
    
    return loss

    
