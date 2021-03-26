import numpy as np

from cvxopt import solvers
from cvxopt import matrix
solvers.options['show_progress'] = False

from library.utilities import *

def Kernel_Ridge_Regression(K,y,lbd=0.005):   #0.005->AUC=[0.72, 0.72, 0.82]   0.0000005->AUC=[0.7, 0.74, 0.83]
  y = np.expand_dims(y,1)
  n = K.shape[0]
  I = np.eye(n)
  a = np.linalg.solve(K+lbd*n*I,y)
  return a, 0




def SVM(K,y,lbd=0.0003):    # 30 -> AUC = [0.72, 0.7, 0.8]    0.3 -> AUC = [0.72, 0.71, 0.8]    0.0003 -> AUC = [0.71, 0.74, 0.83]     0.000003 -> AUC = [0.7, 0.75, 0.83]
  n = K.shape[0]
  C = 1/(2*lbd*n)
  #print('C =',C)
  A = np.concatenate([np.diag(y), -np.diag(y)], axis=0)
  y = np.expand_dims(y,1)
  b = np.vstack([C*np.ones([n,1]), np.zeros([n,1])])
  E = matrix(np.ones([1,n]), tc='d')
  g = matrix(0, tc='d')
  sol = solvers.qp(matrix(2*K, tc='d'),matrix(-2*y, tc='d'),matrix(A, tc='d'),matrix(b, tc='d'),E,g)
  a = np.array(sol['x'])
  interc = y[0] - (K[0]@a)
  return a, interc



def logistic_reg(X,y):
  y = np.expand_dims(y,1)
  n,m = X.shape
  X = np.concatenate([np.ones([n,1]),X],axis=1)
  w = newton(X,y)[-1]
  return w[1:], w[0]


