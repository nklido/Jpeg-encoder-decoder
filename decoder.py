import numpy as np
from scipy import fftpack
import util

class Decoder:
    def __init__(self,n,base):
        self.n=n
        self.base=base

    def performIDCT(self,dct,idct_opt="cust"):
        if(idct_opt=="cust"):
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
        elif(idct_opt=="spicy"):
            im =fftpack.idct(dct)#,norm='ortho')
            #im =fftpack.idct(fftpack.idct(dct.T, norm='ortho').T, norm='ortho')
        else:
            raise ValueError
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

    def performIQuantization(self,f):
        N= self.n
        width = f.shape [0]
        height = f.shape [1]
        #for now assuming that image can be divided perfectly by 8 on both dimensions
        rows = int(height/N)
        cols = int(width/N)

        qu_array = util.getQuantizationArray()
        im =np.zeros((rows*N,cols*N))
        for i in range(rows):
            for j in range(cols):
                im[i*N:i*N+N,j*N:j*N+N] = np.multiply(f[i*N:i*N+N,j*N:j*N+N],qu_array)
        return im

    def decode(self,dct,idct_opt):
        #return self.performIDCT(dct,idct_opt)
        return self.performIDCT(self.performIQuantization(dct),idct_opt)
