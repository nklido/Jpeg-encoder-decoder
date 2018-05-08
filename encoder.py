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

    def performDCT(self,f,dct_opt="custom"):
        if(dct_opt=="custom"):
            dct =np.zeros((self.rows*8,self.cols*8))
            for i in range(self.rows):
                for j in range(self.cols):
                    dct[i*8:i*8+8,j*8:j*8+8] = self.cosTrans(f[i*8:i*8+8,j*8:j*8+8])
        elif(dct_opt=="DCTI"):
            dct =fftpack.dct(fftpack.dct(f,type=1),type=1)
        elif(dct_opt=="DCTII"):
            dct =fftpack.dct(fftpack.dct(f.T, norm='ortho').T, norm='ortho')
        elif(dct_opt=="DCTIII"):
            dct =fftpack.dct(fftpack.dct(f.T,type=3,norm='ortho').T,type=3, norm='ortho')

        else:
            raise ValueError("Invalid dct option. Please use --help for usage.")
        return dct


    def performQuantization(self,arr,quality="high",enableLM=False,enableTM=False):
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
        #Since luminance and texture masking are by default disabled
        #both factors are set by default to 1
        LumMask =np.ones([self.rows,self.cols])
        TexMask =np.ones([self.rows,self.cols])

        #if luminance masking is enabled
        if(enableLM):
            LumMask = self.calculateLumMask(LumMask,arr)
            self.writeLumMaskToFile(LumMask)
        qu_array = util.getQuantizationArray(quality)
        qu =np.zeros((self.rows*8,self.cols*8))
        for i in range(self.rows):
            for j in range(self.cols):
                qu[i*8:i*8+8,j*8:j*8+8] = np.rint(np.divide(arr[i*8:i*8+8,j*8:j*8+8],(LumMask[i,j]*TexMask[i,j])*qu_array))
        return qu

    #accepts an array and for each 8x8 block generates a list of triplets -->(numberofzeros,#bits to repr value,value)
    # result is a list with a list of triplets
    def RLE(self,array):
        result = []
        for i in range(self.rows):
            for j in range(self.cols):
                zeroCount=0
                triplets =[]
                zig = util.zigzagparse(array[i*8:i*8+8,j*8:j*8+8])
                for value in zig.astype(int):
                    if(value!=0):
                        triplets.append((zeroCount,(int(value)).bit_length(),value))
                        zeroCount =0
                    else:
                        zeroCount+=1
                result.append(triplets)
        return result


    def calculateLumMask(self,LumMask,arr):
        FmaxL = 2
        LumMin ,LumMax =90,255

        sum = 0
        for i in range(self.rows):
            for j in range(self.cols):
                sum+=arr[i*8,j*8]
        meanLuminance = sum/(self.rows*self.cols)
        #print("Mean luminance is "+str(meanLuminance))
        Fref =(1/165)*meanLuminance+(5/11)
        for i in range(self.rows):
            for j in range(self.cols):
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
        return LumMask


    def writeLumMaskToFile(self,LumMask):
        file = open('pics/'+self.img_name+"_luminance.txt", 'w')
        for i in range(self.rows):
            for j in range(self.cols):
                file.write("%s\t" % str(LumMask[i,j]))
            file.write("\n")

    def encode(self,img_name,f,dct,quality="high",LM=False,TM=False):
        self.img_name = img_name
        width,height = f.shape [0],f.shape [1]
        #for now assuming that image can be divided perfectly by 8 on both dimensions
        self.rows,self.cols = int(height/self.n),int(width/self.n)


        dct = self.performDCT(f,dct)
        quant_array = self.performQuantization(dct,quality=quality,enableLM=LM,enableTM=TM)

        list_of_triplets = self.RLE(quant_array)
        return list_of_triplets
