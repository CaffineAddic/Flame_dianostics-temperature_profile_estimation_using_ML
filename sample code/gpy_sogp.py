import GPy
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load training data
absorb = scio.loadmat('absorbance_TX_hitemp.mat')['absorb']
paraMat = scio.loadmat('para_TX_hitemp.mat')
Temper = paraMat['Temper']
Temper=np.reshape(Temper,-1)[:,None]
Conc_1 = paraMat['Conc_1']
Conc_1=np.reshape(Conc_1,-1)[:,None]
Conc_2 = paraMat['Conc_2']
Conc_2=np.reshape(Conc_2,-1)[:,None]

# normalization
scalerT = StandardScaler().fit(Temper)
scalerX_1 = StandardScaler().fit(Conc_1)
scalerX_2 = StandardScaler().fit(Conc_2)
Temper=scalerT.transform(Temper)
Conc_1=scalerX_1.transform(Conc_1)
Conc_2=scalerX_2.transform(Conc_2)

# SOGPR model set up (same as ICM)
ks = GPy.kern.RBF(np.shape(absorb)[1])
icm = GPy.util.multioutput.ICM(input_dim=np.shape(absorb)[1],num_outputs=3,kernel=ks)

print(icm)
print('W matrix\n',icm.B.W)
print('\nkappa vector\n',icm.B.kappa)
print('\nB matrix\n',icm.B.B)

m = GPy.models.GPCoregionalizedRegression([absorb,absorb,absorb],[Temper,Conc_1,Conc_2],kernel=icm)
m['.*rbf.var'].constrain_fixed(1.)
# ICM reduce to SOGPR by setting B diagonal
m['.*B.W'][0].constrain_fixed(0.)
m['.*B.W'][1].constrain_fixed(0.)
m['.*B.W'][2].constrain_fixed(0.)
print(m)
print('W matrix\n',icm.B.W)
print('\nkappa vector\n',icm.B.kappa)

# model optimization
m.optimize()
print(m)
print('W matrix\n',icm.B.W)
print('\nkappa vector\n',icm.B.kappa)
print('\nB matrix\n',icm.B.B)

# load test data and normalization
absorb_test = scio.loadmat('absorbance_TXtest_hitemp_noise.mat')['absorb']
paraMat_test = scio.loadmat('para_TXtest_hitemp.mat')
Temper_test = paraMat_test['Temper']
Temper_test=np.reshape(Temper_test,-1)[:,None]
Conc_1_test = paraMat_test['Conc_1']
Conc_1_test=np.reshape(Conc_1_test,-1)[:,None]
Conc_2_test = paraMat_test['Conc_2']
Conc_2_test=np.reshape(Conc_2_test,-1)[:,None]

# prediction of T
newX = absorb_test
newX = np.concatenate((newX,np.zeros((np.shape(absorb_test)[0],1))),1)
noise_dict = {'output_index':newX[:,-1].astype(int)}
ys,var=m.predict(newX,Y_metadata=noise_dict)

Temper_test_=scalerT.inverse_transform(ys)
scio.savemat('sogpResults_T.mat', {'T_pred_sogp': Temper_test_})

# prediction of X1
newX = absorb_test
newX = np.concatenate((newX,np.ones((np.shape(absorb_test)[0],1))),1)
noise_dict = {'output_index':newX[:,-1].astype(int)}
ys,var=m.predict(newX,Y_metadata=noise_dict)

Conc_1_test_=scalerX_1.inverse_transform(ys)
scio.savemat('sogpResults_X1.mat', {'X1_pred_sogp': Conc_1_test_})

# prediction of X2
newX = absorb_test
newX = np.concatenate((newX,2*np.ones((np.shape(absorb_test)[0],1))),1)
noise_dict = {'output_index':newX[:,-1].astype(int)}
ys,var=m.predict(newX,Y_metadata=noise_dict)

Conc_2_test_=scalerX_2.inverse_transform(ys)
scio.savemat('sogpResults_X2.mat', {'X2_pred_sogp': Conc_2_test_})

# export K_M
absorb_T=absorb[99::100,:]
absorb_X1=absorb[909::10,:]
absorb_X2=absorb[990::1,:]
TK=ks.K(absorb_T)
X1K=ks.K(absorb_X1)
X2K=ks.K(absorb_X2)
scio.savemat('sogpResults_gpy_hitemp_model.mat', {'TK': TK, 'X1K': X1K, 'X2K': X2K, 'l': icm.rbf.lengthscale, 'BW': icm.B.W, 'BKappa': icm.B.kappa, 'B': icm.B.B, 'sigma1': m.mixed_noise.Gaussian_noise_0.variance, 'sigma2': m.mixed_noise.Gaussian_noise_1.variance, 'sigma3': m.mixed_noise.Gaussian_noise_2.variance})