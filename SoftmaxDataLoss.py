import numpy as np

def Li_unvec(x,y,W):
    scores = x.dot(W)
    #see http://cs231n.github.io/linear-classify/#softmax, 'Practical issues: Numeric stability' 
    scores-=np.max(scores)
    pass