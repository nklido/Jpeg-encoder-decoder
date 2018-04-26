import numpy as np
from scipy import fftpack
import util

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

    def performDCT(self,f,dct_opt="cust"):
        if(dct_opt=="cust"):
            width = f.shape [0]
            height = f.shape [1]

            #for now assuming that image can be divided perfectly by 8 on both dimensions
            rows = int(height/8)
            cols = int(width/8)

            dct =np.zeros((rows*8,cols*8))
            for i in range(rows):
                for j in range(cols):
                    dct[i*8:i*8+8,j*8:j*8+8] = self.cosTrans(f[i*8:i*8+8,j*8:j*8+8])
        elif(dct_opt=="spicy"):
            dct = fftpack.dct(f)#,norm='ortho')
            #dct =fftpack.dct(fftpack.dct(f.T, norm='ortho').T, norm='ortho')

        else:
            raise ValueError

        return dct



    def performQuantization(self,f):
        width = f.shape [0]
        height = f.shape [1]

        #for now assuming that image can be divided perfectly by 8 on both dimensions
        rows = int(height/8)
        cols = int(width/8)

        qu_array = util.getQuantizationArray()
        qu =np.zeros((rows*8,cols*8))
        for i in range(rows):
            for j in range(cols):
                qu[i*8:i*8+8,j*8:j*8+8] = np.rint(np.divide(f[i*8:i*8+8,j*8:j*8+8],qu_array))
        return qu


    def encode(self,f,dct):
        #return self.performDCT(f,dct)
        return self.performQuantization(self.performDCT(f,dct))



