import numpy as np

class Decoder:
    def __init__(self,n,base):
        self.n=n
        self.base=base

    def myIDCT(self,dct):
        N= self.n
        width = dct.shape [0]
        height = dct.shape [1]

        #for now assuming that image can be divided perfectly by 8 on both dimensions
        rows = int(height/N)
        cols = int(width/N)

        im =np.zeros((rows*N,cols*N))
        for i in range(rows):
            for j in range(cols):
                im[i*N:i*N+N,j*N:j*N+N] = self.cosTransi(dct[i*N:i*N+N,j*N:j*N+N])
        return im

    def cosTransi(self,cosines):
        N = self.n
        fi = np.zeros((N,N))
        for x in range(0,N):
            for y in range(0,N):
                for u in range(0,N):
                    for v in range(0,N):
                        fi[x][y]+=self.base[u,v,x,y]*cosines[u][v]
        return fi