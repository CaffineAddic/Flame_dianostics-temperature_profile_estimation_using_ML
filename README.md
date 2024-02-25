# Flame_dianostics-temperature_profile_estimation_using_ML

In this project we try to estimate the temperature profile in a flame using a multi output regressive processes

*Theory*
--- 
The aim of the project is to estimate temperature profile of a flame of a flame from the absorption spectra of selected gases ($CO_2, H_2O$) using a tuneable laser diode measuring the power spectrum of the incident laser beam and then after passing through the flame energy at particular spectrum will absorbed and measuring the emergent laser beam from the flame to get the absorption spectra.

Now by using the absorption spectrum we are trying to estimate the temperature profile in the flame.

As the we need to estimate we have to use multi output processes and in cases the output may be hetrotropic (Different output dimensions when compared to the input).

To model the problem we are working towards using multi output Gaussian process it's approaches of ICM (Intrinsic Coregionalization Model), LMC (Linear Model of Cogregionalization) and PC (Process Convolution)

I have added a lot of theory and using the paper for reference:

Gaussian process regression for direct laser absorption spectroscopy in complex combustion environments. Opt Express.
 2021 Jun 7;29(12):17926-17939. doi: 10.1364/OE.425662. PMID: 34154064.

Using GpyTorch for implimentations.
