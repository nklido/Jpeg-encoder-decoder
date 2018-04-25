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