import sys
from PIL import Image
import numpy as np
import util
from encoder import Encoder
from decoder import Decoder
import matplotlib.pyplot as plt

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
            n=8
            base = util.calc_dct_base(n)
            enc = Encoder(n,base)
            dct =enc.myDCT(image_arr)


            dec = Decoder(n,base)
            im = dec.myIDCT(dct)



            fig,(ax1, ax2) = plt.subplots(1,2,sharey=True)
            ax1.set_title("Original image")
            ax1.imshow(image_arr,cmap='gray')
            ax2.set_title("After dct -> idct")
            ax2.imshow(im,cmap='gray')

            plt.show()

        except FileNotFoundError:
            print("Please specify an image file..")




