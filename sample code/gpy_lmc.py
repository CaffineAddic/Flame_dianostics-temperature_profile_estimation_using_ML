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

# LMC set up
ks = [GPy.kern.RBF(np.shape(absorb)[1]) for i in range(3)]
icm = GPy.util.multioutput.LCM(input_dim=np.shape(absorb)[1],num_outputs=3,kernels_list=ks)

print(icm)

m = GPy.models.GPCoregionalizedRegression([absorb,absorb,absorb],[Temper,Conc_1,Conc_2],kernel=icm)
m['.*rbf.var'].constrain_fixed(1.)
# constrain if converge to reasonless extrema
# m['mixed_noise.Gaussian_noise_0.variance'].constrain_bounded(0,0.0001)
# m['mixed_noise.Gaussian_noise_1.variance'].constrain_bounded(0,0.0001)
# m['mixed_noise.Gaussian_noise_2.variance'].constrain_bounded(0,0.0001)
print(m)

# model optimization
m.optimize()
print(m)

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
scio.savemat('lmcResults_T.mat', {'T_pred_lmc': Temper_test_})

# prediction of X1
newX = absorb_test
newX = np.concatenate((newX,np.ones((np.shape(absorb_test)[0],1))),1)
noise_dict = {'output_index':newX[:,-1].astype(int)}
ys,var=m.predict(newX,Y_metadata=noise_dict)

Conc_1_test_=scalerX_1.inverse_transform(ys)
scio.savemat('lmcResults_X1.mat', {'X1_pred_lmc': Conc_1_test_})

# prediction of X2
newX = absorb_test
newX = np.concatenate((newX,np.ones((np.shape(absorb_test)[0],1))+np.ones((np.shape(absorb_test)[0],1))),1)
noise_dict = {'output_index':newX[:,-1].astype(int)}
ys,var=m.predict(newX,Y_metadata=noise_dict)

Conc_2_test_=scalerX_2.inverse_transform(ys)
scio.savemat('lmcResults_X2.mat', {'X2_pred_lmc': Conc_2_test_})

# export K_M for Fig. 5
absorb_T=absorb[99::100,:]
absorb_X1=absorb[909::10,:]
absorb_X2=absorb[990::1,:]
TK1=ks[0].K(absorb_T)
X1K1=ks[0].K(absorb_X1)
X2K1=ks[0].K(absorb_X2)
TK2=ks[1].K(absorb_T)
X1K2=ks[1].K(absorb_X1)
X2K2=ks[1].K(absorb_X2)
TK3=ks[2].K(absorb_T)
X1K3=ks[2].K(absorb_X1)
X2K3=ks[2].K(absorb_X2)
scio.savemat('lmcResults_model.mat', {'TK1': TK1, 'X1K1': X1K1, 'X2K1': X2K1, 'TK2': TK2, 'X1K2': X1K2, 'X2K1': X2K2, 'TK3': TK3, 'X1K3': X1K3, 'X2K3': X2K3, 'l1': icm.ICM0.rbf.lengthscale, 'l2': icm.ICM1.rbf.lengthscale, 'l3': icm.ICM2.rbf.lengthscale, 'BW1': icm.ICM0.B.W, 'BW2': icm.ICM1.B.W, 'BW3': icm.ICM2.B.W, 'BKappa1': icm.ICM0.B.kappa, 'BKappa2': icm.ICM1.B.kappa, 'BKappa3': icm.ICM2.B.kappa, 'B1': icm.ICM0.B.B, 'B2': icm.ICM1.B.B, 'B3': icm.ICM2.B.B, 'sigma1': m.mixed_noise.Gaussian_noise_0.variance, 'sigma2': m.mixed_noise.Gaussian_noise_1.variance, 'sigma3': m.mixed_noise.Gaussian_noise_2.variance})