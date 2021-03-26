import numpy as np
import scipy.special as ssp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from library.models import *
from library.utilities import *

def train(K,Y,model,params={'lbd':0.005}):
  if model=='SVM':
      lbd = params['lbd']
      res = SVM(K,2*Y-1, lbd=lbd)
  elif model=='KRR':
    lbd = params['lbd']
    res = Kernel_Ridge_Regression(K,2*Y-1, lbd=lbd)
  return res

def train_check(K,Y,model,params={'lbd':0.005}):
  res = train(K,Y,model,params)
  return res, findBest(K, Y, model, params, n_iter=1,show=1)


def predict(K,coeffs_model,threshold=0.5):
  a,b = coeffs_model
  preds = ssp.expit(K @ a + b).flatten()
  cls = (preds>threshold).astype(int)
  return preds,cls


def showResults(Y_pred_val,Y_val,Y_pred_tr,Y_tr,show=2):
  fpr, tpr, thresholds = roc_curve(Y_val, Y_pred_val)
  fpr_tr, tpr_tr, thresholds_tr = roc_curve(Y_tr, Y_pred_tr)
  #opt_thr = thresholds[np.argmax(tpr - fpr)]
  opt_thr_tr = thresholds_tr[np.argmax(tpr_tr - fpr_tr)]
  opt_thr = opt_thr_tr ###########

  auc = roc_auc_score(Y_val, Y_pred_val)
  auc_tr = roc_auc_score(Y_tr, Y_pred_tr)

  preds_norm = (Y_pred_val - Y_pred_val.mean())/(Y_pred_val.std())
  preds_med = (Y_pred_val - np.median(Y_pred_val))
  cls = (Y_pred_val>0.5)
  cls_norm = (preds_norm>0).astype(int)
  cls_med = (preds_med>0).astype(int)
  acc05 = (cls==Y_val).mean()
  accnorm = (cls_norm==Y_val).mean()
  accmed = (cls_med==Y_val).mean()
  
  Y_pred_cls = (Y_pred_val > opt_thr).astype(int)
  Y_pred_tr_cls = (Y_pred_tr > opt_thr_tr).astype(int)
  miss_clf = (Y_pred_cls != Y_val)
  miss_clf_tr = (Y_pred_tr_cls != Y_tr)
  acc, acc_tr = 1 - miss_clf.mean(), 1 - miss_clf_tr.mean()    # Accuracy
  acc1, acc0 = (Y_pred_cls[Y_val==1].mean()), 1-(Y_pred_cls[Y_val==0].mean())
  if show>0:
    print('Accuracy =',round(acc,2),'  --- ( On class "0" :',round(acc0,2),' - On class "1" :',round(acc1,2),')')
    print('Accuracy by  threshold : [ 1/2 :',acc05,' ---  MEAN :',accnorm,' ---  MED :',accmed,']')
    print('Accuracy Train =',round(acc_tr,2))
    print("AUC =",round(auc,2), '  (Optimal threshold =',opt_thr,')')
    print("AUC Train =",round(auc_tr,2), '  (Optimal threshold =',opt_thr_tr,')')
  if (show==2):
    fig1, axs1 = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
    axs1[0].set_title('ROC curve Test Set')
    axs1[0].plot(fpr,tpr,label='AUC = '+str(round(auc,2)))
    axs1[0].legend(loc='best')
    axs1[1].set_title('ROC curve Train Set')
    axs1[1].plot(fpr_tr,tpr_tr,label='AUC = '+str(round(auc_tr,2)))
    axs1[1].legend(loc='best')
    plt.show()
  return [auc, auc_tr, acc, acc_tr, opt_thr, opt_thr_tr, acc1, acc0, acc05, accnorm,accmed]

def cross_valid(K,Y, model, params={'lbd':0.003}, splits=5, show=0):
  n = Y.shape[0]
  idx = np.random.permutation(n)
  sets = np.array_split(idx, splits)
  results = []
  model_params = []
  rm_sets = []
  for k in range(splits):
    idx_tr = np.delete(np.arange(n),sets[k])
    K_tr = K[idx_tr,:][:,idx_tr]
    Y_tr = Y[idx_tr]
    K_val = K[sets[k],:][:,idx_tr]
    Y_val = Y[sets[k]]
    if model=='SVM':
      lbd = params['lbd']
      res = SVM(K_tr,2*Y_tr-1, lbd=lbd)
    elif model=='KRR':
      lbd = params['lbd']
      res = Kernel_Ridge_Regression(K_tr,2*Y_tr-1, lbd=lbd)
    preds_tr = ssp.expit(K_tr @ res[0] + res[1]).flatten()
    preds_val = ssp.expit(K_val @ res[0] + res[1]).flatten()
    eval = showResults(preds_val, Y_val, preds_tr, Y_tr, show=show)
    results.append(eval)
    model_params.append(res)
    rm_sets.append(np.copy(sets[k]))
  return np.array(results), model_params, rm_sets


def cv_predict(K_train, K_test, Y, dataset=0, model='KRR', params={'lbd':0.0005}, n_iter=10, show=1):
    n_train = Y[ds].shape[0]
    ds = dataset
    res_KRR_CV = []
    params_KRR_CV = []
    rm_sets_KRR_CV = []
    for _ in range(n_iter):
      res_KRR, params_KRR, rm_sets_KRR = cross_valid(K_train[ds],Y[ds], model, params=params, splits=5, show=0)
      res_KRR_CV.append(res_KRR)
      params_KRR_CV+=params_KRR
      rm_sets_KRR_CV.append(rm_sets_KRR)
    res_KRR_CV = np.concatenate(res_KRR_CV,axis=0)
    rm_sets_KRR_CV = np.concatenate(rm_sets_KRR_CV,axis=0)
    avg_KRR_CV = np.mean(res_KRR_CV,axis=0)
    avg_AUC_KRR_CV = avg_KRR_CV[0]
    avg_thr_KRR_CV = avg_KRR_CV[4]
    best = np.argmax(res_KRR_CV[:,0])
    best_KRR_0, best_set_KRR_0, [best_AUC, best_thr] = params_KRR_CV[best], np.delete(np.arange(n_train),rm_sets_KRR_CV[best]), res_KRR_CV[best,[0,4]]
    print('Averaged AUC =',avg_AUC_KRR_CV)
    if(show==1):
      print('Averaged threshold =',avg_thr_KRR_CV)
      print('Best AUC =',best_AUC, ' ( threshold =',best_thr,')')
    preds_KRR, cls_KRR = predict(K_test[ds][:,:][:,best_set_KRR_0],best_KRR_0,best_thr)
    return preds_KRR, cls_KRR


def cv_predict2(K_train, K_test, Y, model='KRR', params={'lbd':0.0005}, n_iter=10,show=1):
    n_train = Y.shape[0]
    res_KRR_CV = []
    params_KRR_CV = []
    rm_sets_KRR_CV = []
    for _ in range(n_iter):
      res_KRR, params_KRR, rm_sets_KRR = cross_valid(K_train,Y, model, params=params, splits=5, show=0)
      res_KRR_CV.append(res_KRR)
      params_KRR_CV+=params_KRR
      rm_sets_KRR_CV.append(rm_sets_KRR)
    res_KRR_CV = np.concatenate(res_KRR_CV,axis=0)
    rm_sets_KRR_CV = np.concatenate(rm_sets_KRR_CV,axis=0)
    avg_KRR_CV = np.mean(res_KRR_CV,axis=0)
    avg_AUC_KRR_CV = avg_KRR_CV[0]
    avg_thr_KRR_CV = avg_KRR_CV[4]
    best = np.argmax(res_KRR_CV[:,0])
    best_KRR_0, best_set_KRR_0, [best_AUC, best_thr] = params_KRR_CV[best], np.delete(np.arange(n_train),rm_sets_KRR_CV[best]), res_KRR_CV[best,[0,4]]
    print('Averaged AUC =',avg_AUC_KRR_CV)
    if (show==1):
      print('Averaged threshold =',avg_thr_KRR_CV)
      print('Best AUC =',best_AUC, ' ( threshold =',best_thr,')')
    preds_KRR, cls_KRR = predict(K_test[:,:][:,best_set_KRR_0],best_KRR_0,best_thr)
    return preds_KRR, cls_KRR, avg_AUC_KRR_CV

def findBest(K_train, Y, model='KRR', params={'lbd':0.0005}, n_iter=10,show=1):
    n_train = Y.shape[0]
    res_CV = []
    params_CV = []
    rm_sets_CV = []
    for _ in range(n_iter):
      res, par, rm_sets = cross_valid(K_train,Y, model, params=params, splits=5, show=0)
      res_CV.append(res)
      params_CV+=par
      rm_sets_CV.append(rm_sets)
    res_CV = np.concatenate(res_CV,axis=0)
    rm_sets_CV = np.concatenate(rm_sets_CV,axis=0)
    avg_CV = np.mean(res_CV,axis=0)
    avg_AUC_CV = avg_CV[0]
    avg_thr_CV = avg_CV[4]
    avg_ACC_CV = avg_CV[[2,8,9,10]]
    best = np.argmax(res_CV[:,0])
    best_model, best_set, [best_AUC, best_thr] = params_CV[best], np.delete(np.arange(n_train),rm_sets_CV[best]), res_CV[best,[0,4]]
    best_ACC = res_CV[best,[2,8,9,10]]
    print('Averaged AUC =',avg_AUC_CV,' -- Averaged ACC =',avg_ACC_CV,'(seuils : opt_thr, 0.5, MOY, MED)')
    if (show==1):
      print('Averaged threshold =',avg_thr_CV)
      print('Best AUC =',best_AUC, ' ( threshold =',best_thr,')')
      print('Best ACC =',best_ACC)
    return avg_CV, [best_model, best_set, best_thr, best_AUC, best_ACC]


def evalModel(kernels={}, model='KRR',datasets=[0,1,2],n_iter=10,lbd_values=0.0000005*10**np.arange(6)):
  res = []
  best_models=[]
  itr,ite = np.arange(2000), np.arange(2000,3000)
  for ds in datasets:
    print('\nDataset ',ds)
    for kern in kernels:
      print('\n'+kern)
      K = kernels[kern]
      K_tr, K_te = split_kernels(K[ds],itr,ite) 
      Y_tr = Y[idx_tr[ds]]
      best_model = 0
      res_max = 0
      auc_max = 0
      lbd_max = 0
      for lbd in lbd_values:
        print('lbd =',lbd)
        avg_CV, best = findBest(K_tr, Y_tr, model=model, n_iter=n_iter, params={'lbd':lbd},show=0)
        auc = avg_CV[0]
        if (auc > auc_max):
          res_max = avg_CV
          auc_max = auc
          lbd_max = lbd
          best_model = best
      print('\nAUC Max =',auc_max,' --  lbd =',lbd_max,'\n')
      res.append([ds,kern,auc_max,lbd_max,res_max])
      best_models.append(best_model)
  return res,best_models


def evaluateKernels(kernels={},model='KRR',datasets=[0,1,2],n_iter=10,lbd_values=0.0000005*10**np.arange(6)):
  res = []
  for ds in datasets:
    print('\nDataset ',ds)
    for kern in kernels:
      print('\nKernel : '+kern)
      Kernel = kernels[kern]
      K_tr, K_te = split_kernels(Kernel,idx_tr[ds],idx_te[ds]) 
      Y_tr = Y[idx_tr[ds]]
      auc_max = 0
      lbd_max = 0
      for lbd in lbd_values:
        print('lbd=',lbd)
        _,_,auc = cv_predict2(K_tr, K_te, Y_tr, model=model, n_iter=n_iter, params={'lbd':lbd},show=0)
        if (auc > auc_max):
          auc_max = auc
          lbd_max = lbd
      print('\nAUC Max =',auc_max,' --  lbd =',lbd_max,'\n')
      res.append([ds,kern,auc_max,lbd_max])
  return res


def cross_valid_no_kernel(K,Y, model, params={'lbd':0.003}, splits=5, show=0):
  n = Y.shape[0]
  idx = np.random.permutation(n)
  sets = np.array_split(idx, splits)
  results = []
  model_params = []
  rm_sets = []
  for k in range(splits):
    idx_tr = np.delete(np.arange(n),sets[k])
    K_tr = K[idx_tr]
    Y_tr = Y[idx_tr]
    K_val = K[sets[k]]
    Y_val = Y[sets[k]]
    if model=='LR':
      res = logistic_reg(K_tr,Y_tr)
    preds_tr = ssp.expit(K_tr @ res[0] + res[1]).flatten()
    preds_val = ssp.expit(K_val @ res[0] + res[1]).flatten()
    eval = showResults(preds_val, Y_val, preds_tr, Y_tr, show=show)
    results.append(eval)
    model_params.append(res)
    rm_sets.append(np.copy(sets[k]))
  return np.array(results), model_params, rm_sets


def combine_preds(preds=[],weights=[]):
  n_preds = len(preds)
  for i,p in enumerate(preds):
    preds[i] = (p-np.median(p))/p.std()
  preds = np.vstack(preds).T
  if len(weights)>0:
    preds = preds*np.array(weights)
  avg = preds.mean(axis=1)
  votes = 2*(preds>0).sum(axis=1)/n_preds - 1
  stds = preds.std(axis=1)
  cls = (avg>0).astype(int)
  return np.vstack([avg,stds,votes,cls]).T