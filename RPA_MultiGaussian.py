#!/usr/bin/env python
import numpy as np

# Debye function for a CGC homopolymer
def gD_CGC(k2):
    ''' Continuous Gaussian Chain '''
    result1 = 1. - k2/3.0 + k2*k2/12.0 - k2*k2*k2/60.0
    result2 = (2/k2**2*(np.exp(-k2)+k2-1))
    return np.where(k2<0.01,result1,result2)

def gD_DGC(k2,_N):
    ''' Discrete Gaussian Chain '''
    gD=0.
    for i in range(0, _N+1):
        for j in range(0, _N+1):
            gD = gD + np.exp(-k2*np.abs(i-j)/_N)
    return gD / (N*N + 2*N + 1)

# Square of smearing function
def Gaussian(k2,a):
    ''' Fourier transform of Gaussian'''
    return np.exp(-k2*a*a/2)

# Pressure: integrand of RPA loop integral
def RPA_Pikernel(_gDGSq,_N,_C):
    Piex_k = _C*_N*_gDGSq/(1. + _N*_C*_gDGSq)
    Piex_k = Piex_k - np.log(1. + _N*_C*_gDGSq)
    return Piex_k

# Chemical potential: integrand of RPA loop integral
def RPA_mukernel(_gDGSq,_N,_C):
    muex_k = _N**2*_gDGSq/(1. + _N*_C*_gDGSq)
    return muex_k

# Intensive Free energy: integrand of RPA loop integral
def RPA_Fkernel(_gDGSq,_N,_C):
    Fex_k = 1. + _N*_C*_gDGSq
    Fex_k = np.log(Fex_k)
    return Fex_k

# RPA observables
def RPA_continuum(a_list,u0_list,_C,_N,UseCGC,_kmin,_kmax,_nkgrid):
    # Generate a large dense 1D mesh of k points
    klist, dk = np.linspace(_kmin, _kmax, _nkgrid, endpoint=True, retstep=True)
    k2list = np.square(klist)

    # build gaussian interactions 
    Gauss_array = np.zeros(klist.size)
    for ng, u0 in enumerate(u0_list):
        prefactor = u0
        Gauss_array = np.add(Gauss_array,prefactor*Gaussian(k2list,a_list[ng]))

    # build the second virial coefficient
    _B2 = 0.
    for ng, u0 in enumerate(u0_list):
        _B2 += u0


    # Form gD*Gaussian for all k
    if UseCGC:
        gDGSq=gD_CGC(k2list)
        np.savetxt("DebyeFunction_CGC.dat",np.transpose([k2list,gDGSq]))
        gDGSq=gDGSq * Gauss_array
        np.savetxt("DebyeFunctionTimesGamma2_CGC.dat",np.transpose([k2list,gDGSq]))
        np.savetxt("Gaussian_DGC_N{}.dat".format(_N),np.transpose([k2list,Gauss_array]))
    else:
        gDGSq=gD_DGC(k2list,_N)
        np.savetxt("DebyeFunction_DGC_N{}.dat".format(_N),np.transpose([k2list,gDGSq]))
        gDGSq=gDGSq * Gauss_array
        np.savetxt("DebyeFunctionTimesGamma2_DGC_N{}.dat".format(_N),np.transpose([k2list,gDGSq]))       
        np.savetxt("Gaussian_DGC_N{}.dat".format(_N),np.transpose([k2list,Gauss_array]))
        
        
        
    #
    FoVig = _C*np.log(_C) - _C
    FoVmft = 0.5*_B2*_C*_C
    FoVex = np.sum(k2list*RPA_Fkernel(gDGSq,_N,_C))/(2.*np.pi)**2*dk
    #
    muig = np.log(_C/_N)
    mumft = _B2*_C*_N
    muex = np.sum(k2list*RPA_mukernel(gDGSq,_N,_C))/(2.*np.pi)**2*dk
    #
    Piig = _C/_N
    Pimft = 0.5*_B2*_C*_C
    Piex = np.sum(k2list*RPA_Pikernel(gDGSq,_N, _C))/(2.*np.pi)**2*dk
    #
    return FoVig+FoVmft+FoVex,FoVig+FoVmft,muig+mumft+muex,muig+mumft,Piig+Pimft+Piex,Piig+Pimft

# System parameters
# T160
a=[7.24315e-01, 2.52640e+00]     # Monomer smearing scale
u0=[1.41476e+01, -3.78941e+01]     # Excluded-volume parameter
# T075
#a=[8.24741e-01, 2.51015e+00]     # Monomer smearing scale
#u0=[2.54908e+01, -2.38570e+01]     # Excluded-volume parameter

N=5      # ONLY FOR DGC
UseCGC = False # Switch between CGC and DGC
log_space = True

# RPA
if UseCGC:
    filename="MultiGauss_RPA_CGC.dat"
else:
    filename="MultiGauss_RPA_DGC_N_{}.dat".format(N)
out=open(filename, 'w')
out.write("# C Pi(RPA) Pi(MFT) mu(RPA) mu(MFT) F(RPA) F(MFT)\n")

if log_space:
    C_values = np.logspace(-6,2,1000)
else:
    C_values = np.linspace(1E-6,10,1000)

for C in C_values.tolist():
    print(C)
    F,F_mft,mu,mu_mft,Pi,Pi_mft = RPA_continuum(a,u0,C,N,UseCGC,0.,15,150) # Max k in Rg units
    out.write("{} {} {} {} {} {} {}\n".format(C,Pi,Pi_mft,mu,mu_mft,F,F_mft))
out.close()
