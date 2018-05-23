import sys
from PIL import Image,ImageChops
from encoder import Encoder
from decoder import Decoder
import matplotlib.pyplot as plt
import numpy as np

def printUsage():
    print(
            "Usage: python jpeg.py [Options] [image]"
            "\nOptions\t\t\tDescription\n"+
            "\t{:10s}".format("-dct")+"select dct mode,from either {custom|DCTI|DCTII|DCTIII}\n"+
            "\t{:10s}".format("-q")+"select quality from either {high|low}\n"+
            "\t{:10s}".format("--LM")+"enable luminance masking\n"+
            "\t{:10s}".format("--help")+"ignores all other options and shows usage)\n"+
            "\t{:10s}".format("--compare")+"ignores LM/TM options and plots all 4 possible combinations of TM,LM,LM&TM and None)"
    )

def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0:
            return False
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0:
            return False
    return True

#TODO :(
def setArg(option,value=""):
    if(option=="dct"):
        global dct_opt
        dct_opt = value
    elif(option=="q"):
        global quality
        quality = value
    elif(option=="compare"):
        global compare
        compare = True
    elif(option=="LM"):
        global LM
        LM= True
    elif(option=="TM"):
        global TM
        TM = True
    elif(option=="help"):
        global help
        help = True

if __name__ =="__main__":
    compare=LM=TM=help = False
    quality = "high"
    dct_opt=  "custom"
    img_name  = ""
    extension = ""
    bandNum = 1
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
            elif "." in arg: #image
                img_name = arg.split(".")[0]
                extension= arg.split(".")[1]

    if(help):
        printUsage()
        sys.exit()

    if(img_name==""):
        raise ValueError("Please specify an .tif image file.")


    image = Image.open('pics/'+img_name+"."+extension)

    #debugging.
    #print(image.format, "%dx%d" % image.size, image.mode)
    image_arr = np.array(image)

    if(image.mode=='L' or image.mode=='P'):
        bandNum = 1
    elif(image.mode=='LA'):
        bandNum = 1
        image_arr = image_arr[:,:,0]
    elif(image.mode=='RGBA'):
        bandNum = 3
        image_arr = image_arr[:,:,:3]
    else:
        print("Image mode was "+image.mode)

    enc = Encoder()
    dec = Decoder()

    if(not compare):
        a  = enc.encode(img_name,image_arr,dct=dct_opt,numOfBands=bandNum,quality=quality,LM=LM,TM=TM)
        im = dec.decode(img_name,(image.size[0],image.size[1]),a,idct_opt=dct_opt,numOfBands=bandNum,quality=quality)
        im = (im-np.amin(im))/(np.amax(im)-np.amin(im)) * 255

        im = Image.fromarray(im)
        im.show()
    else:
        a_non = enc.encode(img_name,image_arr,dct=dct_opt,numOfBands=bandNum,quality=quality,LM=False,TM=False)
        a_lum = enc.encode(img_name,image_arr,dct=dct_opt,numOfBands=bandNum,quality=quality,LM=True,TM=False)
        a_tex = enc.encode(img_name,image_arr,dct=dct_opt,numOfBands=bandNum,quality=quality,LM=False,TM=True)
        a_bth = enc.encode(img_name,image_arr,dct=dct_opt,numOfBands=bandNum,quality=quality,LM=True,TM=True)

        img_non = dec.decode(img_name,(image.size[0],image.size[1]),a_non,idct_opt=dct_opt,numOfBands=bandNum,quality=quality)
        img_lum = dec.decode(img_name,(image.size[0],image.size[1]),a_lum,idct_opt=dct_opt,numOfBands=bandNum,quality=quality)
        img_tex = dec.decode(img_name,(image.size[0],image.size[1]),a_tex,idct_opt=dct_opt,numOfBands=bandNum,quality=quality)
        img_bth = dec.decode(img_name,(image.size[0],image.size[1]),a_bth,idct_opt=dct_opt,numOfBands=bandNum,quality=quality)

        fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,sharey=True)
        ax1.set_title("Original image")
        ax1.imshow(image_arr,cmap='gray')

        ax2.set_title("Static Quantization")
        ax2.imshow(img_non,cmap='gray')

        ax3.set_title("Luminance mask")
        ax3.imshow(img_lum,cmap='gray')

        ax4.set_title("Texture mask")
        ax4.imshow(img_tex,cmap='gray')

        ax5.set_title('Luminance and\nTexture masks')
        ax5.imshow(img_bth,cmap='gray')
        plt.show()