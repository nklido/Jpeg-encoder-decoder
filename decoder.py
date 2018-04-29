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
            im =np.zeros((self.rows*N,self.cols*N))
            for i in range(self.rows):
                for j in range(self.cols):
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

    def performIQuantization(self,f,quality="high",enableLM=False,enableTM=False):
        N= self.n
        LumMask =np.ones([self.rows,self.cols])
        TexMask =np.ones([self.rows,self.cols])

        if(enableLM):
            LumMask= self.readLumMaskFromFile(LumMask)

        qu_array = util.getQuantizationArray()
        im =np.zeros((self.rows*N,self.cols*N))
        for i in range(self.rows):
            for j in range(self.cols):
                im[i*N:i*N+N,j*N:j*N+N] = np.multiply(f[i*N:i*N+N,j*N:j*N+N],(LumMask[i,j]*TexMask[i,j])*qu_array)
        return im

    def readLumMaskFromFile(self,LumMask):
        file = open('pics/'+self.img_name+"_luminance.txt", 'r')
        lines = file.read().split("\t")
        for i in range(32):
            LumMask[i,:]=lines[i*32:i*32+32]
        return LumMask

    def decode(self,name,dct,idct_opt,LM=False,TM=False):
        self.img_name=name
        width,height = dct.shape [0],dct.shape [1]
        #for now assuming that image can be divided perfectly by 8 on both dimensions
        self.rows,self.cols = int(height/self.n),int(width/self.n)

        #return self.performIDCT(dct,idct_opt)
        return self.performIDCT(self.performIQuantization(dct,enableLM = LM,enableTM = TM),idct_opt)
