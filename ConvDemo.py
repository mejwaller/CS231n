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
        
        if(self.P>0):
            self.padded = np.zeros((self.IV.shape[0]+2*self.P,self.IV.shape[1]+2*self.P,self.IV.shape[2]))
            self.padded[self.P:self.padded.shape[0]-self.P,self.P:self.padded.shape[1]-self.P,:]=self.IV
        else:
            self.padded=self.IV
            
      
    #should probably store filters and other params in dict - but this just to ensure O can get 
    #demo workign, not design a fully funtioning convnet framework
    def setFilter(self,filter,num):#set numth filter to filter = throw FilterError if num-1 > K
        if num > self.K:
            raise FilterError
            
        if(num==0):
            self.W0 = filter
            
        if(num==1):
            self.W1=filter
        
        
        
class Error(Exception):#base class
    pass
    
class FilterError(Error):
    pass