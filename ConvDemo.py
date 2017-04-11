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
            self.b0=1
            
        if(num==1):
            self.W1=filter
            self.b1=0
            
    def convolve_naive(self):
        out = np.zeros((3,3,2))
        
        #for i in arange(0,self.padded.shape[0],self.stride):
        #    for j in arange(0,self.padded.shape[1], self.stride):
        #        for k in arange(0,self.padded.shape[3]):
        #            out[i,j,k] = 
        
        for i in xrange(0,self.padded.shape[0]-self.S, self.S):
            print i
            
        out[0,0,0] = np.sum(self.padded[:3,:3,:]*self.W0)+self.b0
        out[0,1,0] = np.sum(self.padded[:3,2:5,:]*self.W0)+self.b0
        out[0,2,0] = np.sum(self.padded[:3,4:7,:]*self.W0)+self.b0
        out[1,0,0] = np.sum(self.padded[2:5,:3,:]*self.W0)+self.b0
        out[1,1,0] = np.sum(self.padded[2:5,2:5,:]*self.W0)+self.b0
        out[1,2,0] = np.sum(self.padded[2:5,4:7,:]*self.W0)+self.b0
        out[2,0,0] = np.sum(self.padded[4:7,:3,:]*self.W0)+self.b0
        out[2,1,0] = np.sum(self.padded[4:7,2:5,:]*self.W0)+self.b0
        out[2,2,0] = np.sum(self.padded[4:7,4:7,:]*self.W0)+self.b0
        
        out[0,0,1] = np.sum(self.padded[:3,:3,:]*self.W1)+self.b1
        out[0,1,1] = np.sum(self.padded[:3,2:5,:]*self.W1)+self.b1
        out[0,2,1] = np.sum(self.padded[:3,4:7,:]*self.W1)+self.b1
        out[1,0,1] = np.sum(self.padded[2:5,:3,:]*self.W1)+self.b1
        out[1,1,1] = np.sum(self.padded[2:5,2:5,:]*self.W1)+self.b1
        out[1,2,1] = np.sum(self.padded[2:5,4:7,:]*self.W1)+self.b1
        out[2,0,1] = np.sum(self.padded[4:7,:3,:]*self.W1)+self.b1
        out[2,1,1] = np.sum(self.padded[4:7,2:5,:]*self.W1)+self.b1
        out[2,2,1] = np.sum(self.padded[4:7,4:7,:]*self.W1)+self.b1
                
        #print "1st area layer 0 is: ", self.padded[:3,:3,0]
        #print "1st area layer 1 is: ", self.padded[:3,:3,1]
        #print "1st area layer 2 is: ", self.padded[:3,:3,2]
        #print "2nd area layer 0 is: ", self.padded[:3,2:5,0]
        #print "2nd area layer 1 is: ", self.padded[:3,2:5,1]
        #print "2nd area layer 2 is: ", self.padded[:3,2:5,2]              
        
        print "out layer 1 is: ", out[:,:,0]
        print "out layer 2 is ", out[:,:,1]
        
        return out

        
        
        
class Error(Exception):#base class
    pass
    
class FilterError(Error):
    pass