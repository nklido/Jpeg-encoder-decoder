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
    im = Image.open('pics/cameraman.tif')

    if(len(sys.argv)==1):
        printUsage()
    else:
        try:
            image = Image.open('pics/'+sys.argv[1])
            image_arr = np.array(image)

            #image_arr = plt.imread('pics/cameraman.tif')
            n=8
            dct_opt = "cust"
            base = util.calc_dct_base(n)

            enc = Encoder(n,base)
            dec = Decoder(n,base)

            a = enc.encode(image_arr,dct_opt)

            #printing the very first block for debugging purps
            df = pd.DataFrame(a[0:8,0:8])
            print(df)

            im = dec.decode(a,dct_opt)



            fig,(ax1, ax2) = plt.subplots(1,2,sharey=True)
            ax1.set_title("Original image")
            ax1.imshow(image_arr,cmap='gray')
            ax2.set_title("After dct -> idct")
            ax2.imshow(im,cmap='gray')

            plt.show()

        except FileNotFoundError:
            print("Please specify an image file..")




