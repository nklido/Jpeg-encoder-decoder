import sys
from PIL import Image
import util
from encoder import Encoder
from decoder import Decoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def printUsage():
    print("Usage: python [Options] jpeg.py [image.tif]")


if __name__ =="__main__":


    if(len(sys.argv)==1):
        printUsage()
    else:
        try:
            #this is an uncommited change
            img_name = sys.argv[1]
            image = Image.open('pics/'+img_name)
            image_arr = np.array(image)

            dct_opt = "cust"
            base = util.calc_dct_base(8)

            enc = Encoder()
            dec = Decoder()

            a = enc.encode(img_name,image_arr,dct_opt,LM=True)

            im = dec.decode(img_name,a,dct_opt,LM=True)

            image_arr2 = np.array(image)
            a = enc.encode(img_name,image_arr,dct_opt)
            im2 = dec.decode(img_name,a,dct_opt)

            fig,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
            ax1.set_title("Original image")
            ax1.imshow(image_arr,cmap='gray')
            ax2.set_title("After dct -> idct")
            ax2.imshow(im2,cmap='gray')
            ax3.set_title("After dct -> idct with Luminance masking")
            ax3.imshow(im,cmap='gray')

            plt.show()

            df = pd.DataFrame(image_arr[0:8,0:8])
            print(df)
            df = pd.DataFrame(im2[0:8,0:8])
            print(df)
            df = pd.DataFrame(im[0:8,0:8])
            print(df)

        except FileNotFoundError:
            print("Please specify an image file..")




