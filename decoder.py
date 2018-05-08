import numpy as np
from scipy import fftpack
import util
import array
class Decoder:
    def __init__(self):
        self.n=8
        self.base=util.calc_dct_base(8)


    def performIDCT(self,dct,idct_opt="custom"):

        N = self.n
        if idct_opt=="custom" :
            im =np.zeros((self.rows*N,self.cols*N))
            for i in range(self.rows):
                for j in range(self.cols):
                    im[i*N:i*N+N,j*N:j*N+N] = self.cosTransi(dct[i*N:i*N+N,j*N:j*N+N])
        elif(idct_opt=="DCTI"):
            im =fftpack.idct(fftpack.idct(dct,type=1),type=1)
        elif(idct_opt=="DCTII"):
            im =fftpack.idct(fftpack.idct(dct.T, norm='ortho').T, norm='ortho')
        elif(idct_opt=="DCTIII"):
            im =fftpack.idct(fftpack.idct(dct.T,type=3, norm='ortho').T,type=3,norm='ortho')
        else:
            raise ValueError("Invalid dct option. Please use --help for usage.")
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

        qu_array = util.getQuantizationArray(quality)
        im =np.zeros((self.rows*N,self.cols*N))
        for i in range(self.rows):
            for j in range(self.cols):
                im[i*N:i*N+N,j*N:j*N+N] = np.multiply(f[i*N:i*N+N,j*N:j*N+N],(LumMask[i,j]*TexMask[i,j])*qu_array)
        return im

    #inverse RLE, accepts a list of triplets of type (numOfzeros,#bits to repr value,value) and returns a 1D array.
    def RLEi(self,list_of_triplets):
        array = np.zeros((32*8,32*8))
        for i in range(32):
            for j in range(32):
                triplets = list_of_triplets[(i*32)+j]
                reverse = np.zeros(64)
                at = 0
                for triplet in triplets:
                    at+=triplet[0]
                    reverse[at]=triplet[2]
                    at+=1
                array[i*8:i*8+8,j*8:j*8+8] =util.zigzagparse(reverse)
        return array

    def readLumMaskFromFile(self,LumMask):
        file = open('pics/'+self.img_name+"_luminance.txt", 'r')
        lines = file.read().split("\t")
        for i in range(32):
            LumMask[i,:]=lines[i*32:i*32+32]
        return LumMask

    def decode(self,name,list_of_triplets,idct_opt,quality="high",LM=False,TM=False):
        self.img_name=name
        quant_arr=self.RLEi(list_of_triplets)
        width,height = quant_arr.shape [0],quant_arr.shape [1]


        #for now assuming that image can be divided perfectly by 8 on both dimensions
        self.rows,self.cols = int(height/self.n),int(width/self.n)
        dct   =  self.performIQuantization(quant_arr,enableLM = LM,enableTM = TM,quality=quality) #revereseQuanti\

        image =  self.performIDCT(dct,idct_opt)
        return image