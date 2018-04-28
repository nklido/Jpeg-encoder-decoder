import numpy as np
from scipy import fftpack
import util

class Encoder:
    def __init__(self):
        self.n=8
        self.base=util.calc_dct_base(8)

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


    def performQuantization(self,arr,quality="medium",enableLM=False,enableTM=False):
        """ Performs quantization on an image array

        :param arr: image array after dct operation
        :param enableLM: if is set to true,the quantization table
                        will be scaled by a LumMask(i,j) which is
                        a factor adapting to the local luminance
                        statistics of the (i,j) 8x8 block.
                        Default value is False.
        :param enableTM: if is set to true,the quantization table
                        will be scaled by a TexMask(i,j) which is
                        a factor adapting to the local texture
                        properties of the (i,j) 8x8 block.
                        Default value is False.
        :param quality: Accepts values "low","medium","high"
                        Default="medim"
        :returns: image array after the operation
        """


        width,height = arr.shape [0],arr.shape [1]

        #for now assuming that image can be divided perfectly by 8 on both dimensions
        rows,cols = int(height/8),int(width/8)

        #Since luminance and texture masking are by default disabled
        #both factors are set by default to 1
        LumMask =np.ones([rows,cols])
        TexMask =np.ones([rows,cols])
        #if luminance masking is enabled
        if(enableLM):
            FmaxL = 2
            LumMin ,LumMax =90,255

            sum = 0
            for i in range(rows):
                for j in range(cols):
                    sum+=arr[i*8,j*8]
            meanLuminance = sum/(rows*cols)
            #print("Mean luminance is "+str(meanLuminance))
            Fref =(1/165)*meanLuminance+(5/11)
            for i in range(rows):
                for j in range(cols):
                    #check if DC is higher than meanLuminance
                    DC = arr[i*8,j*8]
                    if( DC>meanLuminance):
                        #linear model applies
                        LumMask[i,j]= (FmaxL-Fref) * (((DC - meanLuminance)/(LumMax-meanLuminance))+1)
                    else:#DC <= meanLuminance
                        #non-linear approximation applies
                        if(DC>=LumMin):
                            LumMask[i,j]=1
                        elif(DC>=25):
                            LumMask[i,j]=1
                        elif(DC>=15):
                            LumMask[i,j]=1.125
                        else:
                            LumMask[i,j]=1.25

        file = open('pics/'+self.img_name+"_luminance.txt", 'w')
        for i in range(rows):
            for j in range(cols):
                file.write("%s\t" % str(LumMask[i,j]))
            file.write("\n")

        qu_array = util.getQuantizationArray()
        qu =np.zeros((rows*8,cols*8))
        for i in range(rows):
            for j in range(cols):
                qu[i*8:i*8+8,j*8:j*8+8] = np.rint(np.divide(arr[i*8:i*8+8,j*8:j*8+8],(LumMask[i,j]*TexMask[i,j])*qu_array))
        return qu

    def encode(self,img_name,f,dct,LM=False,TM=False):
        self.img_name = img_name
        #return self.performDCT(f,dct)
        return self.performQuantization(self.performDCT(f,dct),enableLM=LM,enableTM=TM)

