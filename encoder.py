import numpy as np


class Encoder:
    def __init__(self,n,base):
        self.n=n
        self.base=base

    def cosTrans(self,f):
        N = self.n
        dct=np.zeros((N,N))
        for u in range(0,N):
            for v in range(0,N):
                for x in range(0,N):
                    for y in range(0,N):
                        dct[u][v] += self.base[u,v,x,y]*f[x][y]
        return dct

    def myDCT(self,f):
        width = f.shape [0]
        height = f.shape [1]

        #for now assuming that image can be divided perfectly by 8 on both dimensions
        rows = int(height/8)
        cols = int(width/8)

        dct =np.zeros((rows*8,cols*8))
        for i in range(rows):
            for j in range(cols):
                dct[i*8:i*8+8,j*8:j*8+8] = self.cosTrans(f[i*8:i*8+8,j*8:j*8+8])
        return dct

