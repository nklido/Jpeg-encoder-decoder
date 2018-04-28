import numpy as np
from scipy import fftpack
import util

class Decoder:
    def __init__(self):
        self.n=8
        self.base=util.calc_dct_base(8)


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

    def performIQuantization(self,f,LM=False,TM=False):


        N= self.n
        width = f.shape [0]
        height = f.shape [1]
        #for now assuming that image can be divided perfectly by 8 on both dimensions
        rows = int(height/N)
        cols = int(width/N)

        LumMask =np.ones([rows,cols])
        TexMask =np.ones([rows,cols])

        if(LM):
            file = open('pics/'+self.img_name+"_luminance.txt", 'r')
            lines = file.read().split("\t")
            for i in range(32):
                LumMask[i,:]=lines[i*32:i*32+32]

        qu_array = util.getQuantizationArray()
        im =np.zeros((rows*N,cols*N))
        for i in range(rows):
            for j in range(cols):
                im[i*N:i*N+N,j*N:j*N+N] = np.multiply(f[i*N:i*N+N,j*N:j*N+N],(LumMask[i,j]*TexMask[i,j])*qu_array)
        return im

    def decode(self,name,dct,idct_opt,LM=False,TM=False):
        self.img_name=name
        #return self.performIDCT(dct,idct_opt)
        return self.performIDCT(self.performIQuantization(dct,LM,TM),idct_opt)
