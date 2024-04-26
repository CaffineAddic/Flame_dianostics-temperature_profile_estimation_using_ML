# Multi-output Gaussian process regression for direct laser absorption spectroscopy in complex combustion environments README

by Weitian Wang

Supplementary code to *Multi-output Gaussian process regression for direct laser absorption spectroscopy in complex combustion environments* by Wang et al. 

The three python scripts implements SOGPR (*gpy_sogp.py*), ICM (*gpy_icm.py*) and LMC (*gpy_lmc.py*) models introduced in the paper, respectively. After installing the GPy module properly (http://github.com/SheffieldML/GPy), each script can be run independently to obtain the corresponding results. 

The four .mat files are simulated training and testing spectral data and their corresponding parameters. *absorbance_TX_hitemp.mat* is the training spectra, while *para_TX_hitemp.mat* is the parameters for training spectra. *absorbance_TXtest_hitemp_noise.mat* is the test spectra, while *para_TXtest_hitemp.mat* is the parameters for test spectra. Note that variable *Conc_1* in parameter .mat corresponds to H2O concentration, and variable *Conc_2* corresponds to CO2 concentration. 