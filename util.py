import numpy as np
import math


#factor used when calculating basis function
def a(u,N):
    if u == 0:
        return math.sqrt(1/N)
    else:
        return math.sqrt(2/N)

#calculae dct basis functions
def calc_dct_base(N):
    base = np.zeros((N,N,N,N))
    for u in range(0,N):
        for v in range(0,N):
            for x in range(0,N):
                for y in range(0,N):
                    base[u,v,x,y] = a(u,N)*a(v,N)*math.cos((math.pi*(2*x+1)*u)/(2*N))*math.cos((math.pi*(2*y+1)*v)/(2*N))
    return base


def getQuantizationArray(quality = "high"):
    #example higher quality/lower value quantization array
    if(quality=="high"):
        return np.array([[6,4,4,6,10,16,20,24],
                [5,5,6,8,10,23,24,22],
                [6,5,6,10,16,23,28,22],
                [6,7,9,12,20,35,32,25],
                [7,9,15,22,27,44,41,31],
                [10,14,22,26,32,42,45,37],
                [20,26,31,35,41,48,48,40],
                [29,37,38,39,45,40,41,40]])
    elif(quality=="low"):
    #example lower quality/high value quantization array
        return np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                     [14,17,22,29,51,87,80,62],
                     [18,22,37,56,68,109,103,77],
                     [24,35,55,64,81,104,113,92],
                     [49,64,78,87,103,121,120,101],
                     [72,92,95,98,112,100,103,99]])
    else:
        raise ValueError('Invalid quality parameter...')

def myRLE(array):
    arr=np.zeros(array.size)
    up=False
    i,j,k = 0,1,1
    arr[0] = array[0,0]
    #arr[array.size-1] = array[array.shape[0]-1,array.shape[1]-1]
    while(i<array.shape[0] and i<array.shape[1]):
        arr[k]=array[i,j]
        k+=1
        if(j==array.shape[1]-1 and up): #right side moving up
            i+=1
            up=False
            continue
        elif(j==0 and not up):          #left side moving down
            if(i==array.shape[0]-1):
                j+=1
            else:
                i+=1
            up=True
            continue
        elif(i==0 and up):              #top side moving up
            if(j==array.shape[1]-1):
                i+=1
            else:
                j+=1
            up=False
            continue
        elif(i==array.shape[0]-1 and not up):#bottom side moving down
            j+=1
            up=True
            continue

        #if all checks are clear move diagonally depending on momentum(up~down)
        if(up):
            i=i-1
            j=j+1
        else:
            i=i+1
            j=j-1
    return arr
