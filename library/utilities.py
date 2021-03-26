import numpy as np
import pandas as pd
import scipy.special as ssp
import matplotlib.pyplot as plt
import scipy.sparse as sspa
import itertools


idx_tr = [np.arange(2000),np.arange(2000,4000),np.arange(4000,6000)]
idx_te = [np.arange(6000,7000),np.arange(7000,8000),np.arange(8000,9000)]

Ytr0 = pd.read_csv('data/Ytr0.csv')
Ytr1 = pd.read_csv('data/Ytr1.csv')
Ytr2 = pd.read_csv('data/Ytr2.csv')
Y_data = [Ytr0.Bound, Ytr1.Bound, Ytr2.Bound]
Y = np.concatenate(Y_data, axis=0)

def split_kernels(K,idx_tr,idx_te):
  K_tr = K[idx_tr,:][:,idx_tr]
  K_te = K[idx_te,:][:,idx_tr]
  return K_tr, K_te


def backtracking(t0,v,f,gradf_v,delta,alpha,beta):
  MAX_STEPS=200
  def backtrack(t,step):
    if (step==MAX_STEPS) or (f(v+t*delta)<f(v)+alpha*t*gradf_v.T@delta):
        return t
    step+=1
    return backtrack(beta*t,step)
  return backtrack(t0,0)

def newton(X,y,t=0.1,eps=1e-6,alpha=0.2,beta=0.5):
    n,m = X.shape
    # On crée la fonction objectif ((-) la log-vraisemblance) à minimiser
    # ainsi que son gradient et sa hessienne
    f = lambda w : -np.sum(y*(X@w) - np.log(1+np.exp(X@w)))  # Opposé de la Log-vraisemblance de l'échantillon (x_1,y_1),...,(x_n,y_n)
    gradf = lambda w : -np.sum((y - 1/(1+np.exp(-X@w)))*X,axis=0,keepdims=True).T 
    hessf = lambda w : (X/(1+np.exp(X@w))).T @ (X/(1+np.exp(-X@w)))
    MAX_STEPS = 100
    step = 0
    w0 = np.zeros([m,1])
    v = [w0]
    criteres = []
    while (step<MAX_STEPS):
      step+=1
      gradf_v = gradf(v[-1])
      delta_vnt = - np.linalg.solve(hessf(v[-1]), gradf_v) # calcul du pas optimal de Newton
      crit = -gradf_v.T@delta_vnt # calcul du critère de Newton
      criteres.append(np.float(crit))
      if crit<=2*eps: # critère d'arrêt
        break
      t_b = backtracking(1.,v[-1],f,gradf_v,delta_vnt,alpha,beta) # recherche longueur du pas optimal par backtracking line search
      v.append(np.copy(v[-1]) + t_b*delta_vnt) # mise à jour du v optimal
    return np.array(v)



# Fonctions pour l'encodage des séquences en vecteurs numériques

def divideSeq(seq, size=3, step=1):
    return [seq[x:x+size].upper() for x in range(0,len(seq) - size + 1,step)]
def SeqstoLists(seqs, size=3, step=1):
  return [divideSeq(seq, size, step) for seq in seqs]
def SeqstoStrs(seqs, size=3, step=1):
  return [' '.join(divideSeq(seq, size, step)) for seq in seqs]

codons = { 
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W', 
    }
aa_to_num = {aa:i for i,aa in enumerate(sorted(set(codons.values())))}
codons_to_aa = {cod:aa_to_num[aa] for cod,aa in codons.items()}
codons_to_num = {cod:i for i,cod in enumerate(sorted(codons.keys()))}

def seq_to_num(seqs,to='aa',OH=True,size_sent=1):
  if to=='aa':
    L = [divideSeq(seq, 3,1) for seq in seqs]
    nb = len(aa_to_num)
    L = [[codons_to_aa[cod] for cod in seq] for seq in L]
  else:
    size = len(list(to.keys())[0].split(' ')[0])
    L = [divideSeq(seq, size,1) for seq in seqs]
    if size_sent>1:
      L = [[' '.join(l[i:i+size_sent]) for i in range(len(l)-size_sent)] for l in L]
    nb = len(to)
    L = [[to[cod] for cod in seq] for seq in L]
  L = np.array(L).astype(int)
  if OH:
    n,m = L.shape
    L += np.arange(m)*nb
    X = sspa.csr_matrix((np.ones(n*m),L.flatten(),np.arange(n+1)*m), shape=(n,m*nb))
    return X
    # Matrice dense (numpy)
    #X = np.zeros([n,m,nb])
    #for i in range(n):
      #X[i,np.arange(m),L[i]] = 1
    #return X.reshape(n,m*nb)
  else:
    return L

def ps(X):
  return X @ X.T

def norm2(X):
  return X.power(2).sum(axis=1)

def dist2(X):
  n2 = norm2(X)
  p = ps(X)
  return n2 + n2.T - 2*p.todense()

def gauss_kernel(X,gamma=0):
  m = X.shape[0]
  d2 = dist2(X)
  if gamma==0:
    gamma = 1/(m * d2.var())
  return np.array(np.exp(-gamma * d2))


def dict_of_words(size):
  words = {''.join(v):i for i,v in enumerate(itertools.product('ACGT', repeat=size))}
  return words

def dict_of_sentences(words, size):
  sentences = {' '.join(s):i for i,s in enumerate(itertools.product(list(words.keys()),repeat=size))}
  return sentences

def encode(seqs,size=3,size_sent=1,tfidf=False):
  words = dict_of_words(size)
  if size_sent>1:
    words = dict_of_sentences(words, size_sent)
  nb_cod = len(words)
  X = seq_to_num(seqs,to=words,OH=False,size_sent=size_sent)
  data, idx, idxptr = [], [], [0]
  n = X.shape[0]
  for s in X:
    occ = np.bincount(s,minlength=len(words))
    pos = np.where(occ>0)[0]
    data.append(occ[pos])
    idx.append(pos)
    idxptr.append(pos.shape[0])
  data = np.concatenate(data)
  idx = np.concatenate(idx)
  idxptr = np.cumsum(idxptr)
  X = sspa.csr_matrix((data,idx,idxptr),shape=(n,nb_cod))
  if tfidf:
    tot1 = np.array(X.sum(axis=0))[0]
    tot2 = np.array((X>0).sum(axis=0))[0]
    zeros = (tot1 == 0)
    inv1 = sspa.diags(np.where(zeros,0,1/tot1))
    inv2 = sspa.diags(np.where(zeros,0,np.log(n/tot2)))
    X = X @ inv1
    X = X @ inv2
  #X = np.vstack([np.bincount(s,minlength=nb_cod) for s in X])
  return X


