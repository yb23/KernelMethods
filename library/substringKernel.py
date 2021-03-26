import numpy as np
import pandas as pd
from time import time
from numba import jit

@jit(nopython=True)
def substringB2(x,ys,deltas,kmax=3,lbd=0.9):
  n,len_y = ys.shape
  len_x = x.shape[0]
  B = np.zeros((kmax+1,len_x+1,len_y+1,n))
  B[0] = np.ones((len_x+1,len_y+1,n))
  for k in range(1,kmax+1):
    for i in range(k,len_x+1):
      for j in range(k,len_y+1):
        B[k,i,j] = lbd*(B[k,i-1,j]+B[k,i,j-1]) + lbd**2*(deltas[i-1,j-1]*B[k-1,i-1,j-1] - B[k,i-1,j-1])
  return B

@jit(nopython=True)
def substringK2(x,ys,B,deltas,k=3,lbd=0.9):
  n,len_y = ys.shape
  len_x = x.shape[0]
  K = np.zeros((len_x+1,n))
  for i in range(1,len_x+1):
    sum = (deltas[i-1] * B[k-1,i,1:]).sum(axis=0)
    K[i] = K[i-1] + lbd**2 * sum
  return K

@jit(nopython=True)
def substringKernel2(x,ys,deltas,k=3,lbd=0.9):
  B = substringB2(x,ys,deltas,k,lbd)
  K = substringK2(x,ys,B,deltas,k,lbd)
  return K[-1]