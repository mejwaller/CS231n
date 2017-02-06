import numpy as np

def Li_unvec(x,y,W):
    scores = x.dot(W)
    #see http://cs231n.github.io/linear-classify/#softmax, 'Practical issues: Numeric stability' 
    scores-=np.max(scores)
    p = np.exp(scores)/np.sum(np.exp(scores))
    loss_i = np.log(p[y]) #p[y] are the 'correct' class scores
    return loss_i
    
def SoftmaxLoss(X,y,W):
    scores = X.dot(W)
    pass
    
