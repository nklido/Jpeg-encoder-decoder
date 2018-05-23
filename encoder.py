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
            dct =fftpack.dct(fftpack.dct(f.T,norm='ortho').T, norm='ortho')
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
        if(enableTM):
            TexMask = self.calculateTexMask(TexMask,arr)
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

        arr = (arr-np.amin(arr))/(np.amax(arr)-np.amin(arr)) * 255
        sum = 0
        for i in range(self.rows):
            for j in range(self.cols):
                sum+=arr[i*8,j*8]
        meanLuminance = sum/(self.rows*self.cols)
        Fref =(1/165)*meanLuminance+(5/11)
        for i in range(self.rows):
            for j in range(self.cols):
                #check if DC is higher than meanLuminance
                DC = arr[i*8,j*8]
                if( DC>meanLuminance):
                    #linear model applies
                    LumMask[i,j]= (FmaxL-Fref) * (((DC - meanLuminance)/(LumMax-meanLuminance)))+1
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

    def calculateTexMask(self,TexMask,arr):
        arr = (arr-np.amin(arr))/(np.amax(arr)-np.amin(arr)) * 255 #normalize

        #classification params
        m1 = 125
        m2 = 900
        a1,b1 = 2.3,1.6
        a2,b2 = 1.4,1.1
        g =4
        k = 290

        # params texMask in case of Texture
        tmax,tmin = 1800,290
        fmaxt     = 2.25

        # points relative to [0,0]
        relDC  = [(0,0)]
        relLow = [(0,1),(1,0),(0,2),(2,0),(1,1)]
        relEdge = [(0,3),(0,4),(0,5),(0,6),(3,0),(4,0),(5,0),(6,0),(2,1),(1,2),(2,2),(3,3)]
        relHigh = [(i,j) for i in range(8) for j in range(8) if(i,j) not in relDC+relLow+relEdge]

        for i in range(self.rows):
            for j in range(self.cols):
                block_class = ''

                DC = arr[i*8,j*8]
                L = 0
                E = 0
                H = 0
                for point in relLow:
                    L+=arr[i*8+point[0],j*8+point[1]]
                for point in relEdge:
                    E+=arr[i*8+point[0],j*8+point[1]]
                for point in relHigh:
                    H+=arr[i*8+point[0],j*8+point[1]]

                ti = E+H #texture indicator

                #edge indicators
                edg1 = (L+E)/H
                edg2 = L/E

                if ti > m1:# A
                    if ti > m2: #B
                        if (edg2 >= a2 and edg1>=b2) or (edg2>=b2 and edg1>=a2) or edg1>=g: #C2
                            block_class = 'edge'
                        else:
                            block_class = 'texture'
                    else:
                        if (edg2 >=a1 and edg1>=b1) or (edg2 >=b1 and edg1>=a1) or edg1>=g: #C1
                            block_class = 'edge'
                        else:
                            if ti>k: #D
                                block_class = 'texture'
                            else:
                                block_class = 'plain'
                else:
                    block_class = 'plain'

                if block_class=='plain':
                    TexMask[i,j]=1
                elif block_class=='texture':
                    TexMask[i,j] =(fmaxt-1)*((ti-tmin)/(tmax-tmin))+1
                elif block_class=='edge':
                    if L+E <= 400:
                        TexMask[i,j]=1.125
                    else:
                        TexMask[i,j]=1.25
        return TexMask

    #JUST for debugging
    def writeLumMaskToFile(self,LumMask):
        file = open('pics/'+self.img_name+"_luminance.txt", 'w')
        for i in range(self.rows):
            for j in range(self.cols):
                file.write("%s\t" % str(LumMask[i,j]))
            file.write("\n")

    def encode(self,img_name,img_array,dct,numOfBands=1,quality="high",LM=False,TM=False):
        self.img_name = img_name
        width,height = img_array.shape [0],img_array.shape [1]

        #for now assuming that image can be divided perfectly by 8 on both dimensions
        self.rows,self.cols = int(height/self.n),int(width/self.n)

        if (numOfBands==1):
              data = self.RLE(self.performQuantization(self.performDCT(img_array,dct),quality=quality,enableLM=LM,enableTM=TM))
        else:
            data=[]
            for i in range(numOfBands):
                data.append(self.RLE(self.performQuantization(self.performDCT(img_array[:,:,i],dct),quality=quality,enableLM=LM,enableTM=TM)))

        return data

