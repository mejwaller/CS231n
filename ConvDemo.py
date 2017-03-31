import numpy as np

class net:
    def __init__(self,wid,hgt,dpth,fltrsize,stride,padding,nfilters,inimg):
        self.W=wid
        self.H=hgt
        self.D=dpth
        self.F=fltrsize
        self.S=stride
        self.P=padding
        self.K=nfilters
        self.IV=inimg
        
    def setFilter(self,filter,num):#set numth filter to filter = throw FilterError if num-1 > K
        if num > self.K:
            raise FilterError
        
        
class Error(Exception):#base class
    pass
    
class FilterError(Error):
    pass