import sys
from PIL import Image
import util
from encoder import Encoder
from decoder import Decoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def printUsage():
    print(
            "Usage: python jpeg.py [Options] [image.tif]"
            "\nOptions\n"+
            "\t{:9s}".format("-dct")+"select dct mode,from either {custom|DCTI|DCTII|DCTIII}\n"+
            "\t{:9s}".format("-q")+"select quality from either {high|low}\n"+
            "\t{:9s}".format("--LM")+"enable luminance masking\n"+
            "\t{:9s}".format("--help")+"ignores all other options and shows usage)\n"+
            "\t{:9s}".format("-debug")+"ignores all other options and executs fixed statements(debugging only.)"
    )


def setArg(option,value=""):
    if(option=="dct"):
        global dct_opt
        dct_opt = value
    elif(option=="q"):
        global quality
        quality = value
    elif(option=="debug"):
        global debug
        debug = True
    elif(option=="LM"):
        global LM
        LM= True
    elif(option=="help"):
        global help
        help = True

if __name__ =="__main__":
    debug=LM=help = False
    quality = "high"
    dct_opt="custom"
    img_name =""
    if(len(sys.argv)==1):
        printUsage()
        sys.exit()
    else:
        for index,arg in enumerate(sys.argv[1:],1):
            if "--" in arg: #one word arguments
                setArg(arg.replace("--",""))
            elif "-" in arg: #option arguments
                option = arg.replace("-","")
                value  = sys.argv[index+1]
                setArg(option,value)
            elif ".tif" in arg: #image
                img_name = arg

    if(help):
        printUsage()
        sys.exit()
    if(not debug):
        if(img_name==""):
            raise ValueError("Please specify an .tif image file.")
        try:
            image = Image.open('pics/'+img_name)
            image_arr = np.array(image)
            enc = Encoder()
            dec = Decoder()
            a = enc.encode(img_name.replace(".tif",""),image_arr,dct_opt,quality=quality,LM=LM)
            im = dec.decode(img_name.replace(".tif",""),a,dct_opt,quality=quality,LM=LM)

            fig,(ax1,ax2) = plt.subplots(1,2,sharey=True)
            ax1.set_title("Original image")
            ax1.imshow(image_arr,cmap='gray')

            title = "Image after DCT/IDCT type : {:s}\nLuminance Mask:{:s}\nTexture Mask:{:s}\nQuality chosen:{:s}".format(dct_opt,str(LM),str(False),quality)
            ax2.set_title(title)
            ax2.imshow(im,cmap='gray')
            plt.show()

        except:
            FileNotFoundError("Cannot open file specified.")


    if(debug):
        try:
            #this is an uncommited change
            img_name = "cameraman"
            image = Image.open('pics/cameraman.tif')
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
        except FileNotFoundError:
            print("Please specify an image file..")




