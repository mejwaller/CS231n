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