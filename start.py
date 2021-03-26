from library.models import *
from library.code import *
from library.utilities import *

Str0 = pd.read_csv('data/Xtr0.csv')
Str1 = pd.read_csv('data/Xtr1.csv')
Str2 = pd.read_csv('data/Xtr2.csv')
Ste0 = pd.read_csv('data/Xte0.csv')
Ste1 = pd.read_csv('data/Xte1.csv')
Ste2 = pd.read_csv('data/Xte2.csv')

Str = [Str0, Str1, Str2]
Ste = [Ste0, Ste1, Ste2]
S = [list(Str[i].seq.values)+list(Ste[i].seq.values) for i in range(3)]

print('Calcul des noyaux...')
Ks8 = [gauss_kernel(encode(seqs,size=8),gamma=0.005) for seqs in S]
Ks7 = [gauss_kernel(encode(seqs,size=8),gamma=0.005) for seqs in S]
Kg32 = [gauss_kernel(encode(seqs,size=2,size_sent=3),gamma=0.005) for seqs in S]
Kg33 = [gauss_kernel(encode(seqs,size=3,size_sent=3),gamma=0.005) for seqs in S]
print('Noyaux calculés')

print('Entraînement des modèles...')
res01 = train(Ks7[0][:2000,:][:,:2000],Y[:2000],'KRR',params={'lbd':0.0005})
preds01,_ = predict(Ks7[0][2000:,:][:,:2000],res01,threshold=0.5)
res02 = train(Ks8[0][:2000,:][:,:2000],Y[:2000],'KRR',params={'lbd':0.0005})
preds02,_ = predict(Ks8[0][2000:,:][:,:2000],res02,threshold=0.5)

res11 = train(Kg32[1][:2000,:][:,:2000],Y[2000:4000],'KRR',params={'lbd':0.0005})
preds11,_ = predict(Kg32[1][2000:,:][:,:2000],res11,threshold=0.5)
res12 = train(Kg33[1][:2000,:][:,:2000],Y[2000:4000],'KRR',params={'lbd':0.0005})
preds12,_ = predict(Kg33[1][2000:,:][:,:2000],res12,threshold=0.5)

res21 = train(Ks7[2][:2000,:][:,:2000],Y[4000:6000],'KRR',params={'lbd':0.0005})
preds21,_ = predict(Ks7[2][2000:,:][:,:2000],res21,threshold=0.5)
res22 = train(Ks8[2][:2000,:][:,:2000],Y[4000:6000],'KRR',params={'lbd':0.0005})
preds22,_ = predict(Ks8[2][2000:,:][:,:2000],res22,threshold=0.5)
print('Entraînement terminé')

vals0 = combine_preds(preds=[preds01,preds02])
vals1 = combine_preds(preds=[preds11,preds12])
vals2 = combine_preds(preds=[preds21,preds22])

cls = np.concatenate([vals0[:,3],vals1[:,3],vals2[:,3]]).astype(int)

filename = 'predictions_start.csv'
toKaggle = pd.DataFrame(cls)
toKaggle.to_csv(filename,index_label='Id',header=['Bound'])
print('Prédiction effectuée : '+filename)
