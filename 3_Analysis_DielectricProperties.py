# -*- coding: utf-8 -*-
"""
Created on Fri Jan  19 13:43:40 2018

@author: Cota
"""

import numpy as np
import matplotlib.pyplot as plt
import Fitting_Functions as FU2


from scipy.optimize import curve_fit
from scipy.optimize import leastsq

from matplotlib.backends.backend_pdf import PdfPages


#%%
#Defining a residual function	
def 	residual1(p): #With no conduction
	S, tau, alpha = p
	return np.sqrt(np.power(epsR-FU2.Cole(xN, S, tau, alpha).real,2)+np.power(epsI-FU2.Cole(xN, S, tau, alpha).imag,2))

#Defining a residual function	
def 	residual2(p): #With conduction (Ions)
	S, tau, alpha, sigma = p
	return np.sqrt(np.power(epsR-FU2.ColeCon(xN, S, tau, alpha, sigma).real,2)+np.power(epsI-FU2.ColeCon(xN, S, tau, alpha, sigma).imag,2))


#%%
########     IMPORT DATA     ########

date = '20171017'
#File that contains the names of the samples
namesData = 'Sample_Names_' + date + '.txt'

#Reading the names of the samples 
epsName = np.loadtxt(namesData,comments='#',dtype='str')

#Number of samples which epsilon has been calculated and stored from load_scat2perm_example (Check script!)  
Nsamples=len(epsName)



#File that contains all the experimentally calculated permittivities
epsData = 'Permittivity_Curves_20171017.dat'

#Reading the experimentally calculated permitivities
eps = np.loadtxt(epsData,comments='#',delimiter='\t')

#Defining the frequencies by array "x".
x=eps[:,0]

########################################


#%%
##########     ION CONCENTRATION AND SOLUTION DENSITY      ##########

#These data are important to calculate the reduction of the dielectric 
#response related with the effect of dilution of the solvent.

#Concentration of HCL in H2O given in mol/liter 
Nn=[	0.1, 0.1, 
	0.2, 0.2,
	0.3, 0.3,
	0.4, 0.4,
	0.5, 0.5,
	0.6, 0.6,
	0.7, 0.7,
	0.8, 0.8,
	0.9, 0.9,
	1.0, 1.0]

#Measured density of HCL in H2O given in gr/liter 
Rho=[	0998.90, 0998.90,
	1000.70, 1000.70, 
	1002.50, 1002.50, 
	1004.40, 1004.40, 
	1006.20, 1006.20, 
	1008.10, 1008.10, 
	1009.80, 1009.80, 
	1011.60, 1011.60, 
	1013.40, 1013.40, 
	1015.20, 1015.20]




#%%
#Define few arrays that will serve to store the fitting parameters

#Parameters array
Parameters=np.empty((Nsamples-5,7))

#The information of the two following arrays will also be contained 
#in the array called Parameters

#Depolarization array
Depo=np.empty((Nsamples-5,3))
#Conduction array
Condu=np.empty((Nsamples-5,3))



########################
#####    FITTING PROCESS
########################

print ('\nDielectric properties of HCl:H2O solutions')
print ('using a Cole-Cole relaxation mode\n')

fmt = '{:8} | {:10} | {:10} | {:10} | {:10} | {:10} | {:10}'
header = fmt.format('conc (M)', 'rho (gr/l)', 'S_n', 'S', 'tau (ps)',  'alpha', 'kappa (S/m)')
print('{}\n{}'.format(header, '-'*len(header)))

fmt = '{:8.1f} | {:10.2f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f}'


for i in range(Nsamples):

    if(i==1):

        epsReal, epsImag = eps[:,1+2*i], eps[:,2+2*i]

        #Fit in the spacetral range of 1e9 -- 50e9
        xN = x[112:]
        epsR = epsReal[112:]
        epsI = epsImag[112:]
        

        #Inicial Values
        p0=[73.0, 8.7e-12, 0.00]

        #Least square value. Minimize the value of X^2
        par, cov, info, errmsg, ier = leastsq(residual1, p0, full_output=1)

        #Reduced X^2 
        reducedX2=(((residual1(par))**2).sum())/(len(xN)-len(par))
        #Covariance
        covar=cov*reducedX2
        #Parameters error
        parerr=np.array([np.sqrt(covar[l][l]) for l in range(len(par))])


        header = fmt.format(0.0, 997.00, 73.1727, par[0], par[1]*1e12, 0.0, 0.0000)
        print(header)



    if(i>4):

        epsReal, epsImag = eps[:,1+2*i], eps[:,2+2*i]

        #Fit in the spacetral range of 1e9 -- 50e9
        xN = x[112:]
        epsR = epsReal[112:]
        epsI = epsImag[112:]
            

        ###Fitting only the imaginary to estimate better the initial 
        ###values for the full analysis of the data"""
        ImagpoptCon, ImagpcovCon = curve_fit(FU2.ImagColeCon, x[:290], epsImag[:290], p0=[epsReal[112]-5.8770, 8.5e-12, 0.01, 8.0])


        #Inicial Values
        p0=[ImagpoptCon[0], ImagpoptCon[1], ImagpoptCon[2], ImagpoptCon[3]]

        #Least square value. Minimize the value of X^2
        par, cov, info, errmsg, ier = leastsq(residual2, p0, full_output=1)
          
        #Reduced X^2 
        reducedX2=(((residual2(par))**2).sum())/(len(xN)-len(par))
        #Covariance
        covar=cov*reducedX2
        #Parameters error
        parerr=np.array([np.sqrt(covar[l][l]) for l in range(len(par))])


        #Calculate the effect of dilution of the solvent
        Sn = FU2.SnH(Nn[i-5], Rho[i-5], 73.1727, 0.0)

        header = fmt.format(Nn[i-5], Rho[i-5], Sn[0], par[0], par[1]*1e12, par[2], par[3])
        print(header)

        DeltaS = [Sn[0]-par[0], np.sqrt(((Sn[1])**2)+((parerr[0])**2))]

        Depo[i-5] = [Nn[i-5]] + DeltaS

        Condu[i-5] = [Nn[i-5]] + [par[3], parerr[3]]

        par[1] = par[1]*1e12
        Parameters[i-5] = np.concatenate((np.array([Nn[i-5], Rho[i-5], Sn[0]]), par), axis = 0)


print ('\n')




#%%
##############
####   Exports the dielctric properties of the samples 
####   aka fitting parameters.
##############

annot = 'Measured depolarization of HCl in H2O measured on {}\nConc\tDepo\tdDepo'.format(date)
np.savetxt('Fit_Depolarization_%s.txt' % date, Depo, fmt='%.4f', delimiter='\t', header=annot)


annot = 'Measured conductivity of HCl in H2O measured on {}\nConc\tkappa\tdkappa'.format(date)
np.savetxt('Fit_Conductivity_%s.txt' % date, Condu, fmt='%.4f', delimiter='\t', header=annot)


annot = 'Cole-Cole parameters of HCl in H2O measured on {}\nConc\tRho\tSn\tS\ttau\talpha\tkappa'.format(date)
np.savetxt('Fit_Parameters_%s.txt' % date, Parameters, fmt='%.5f', delimiter='\t', header=annot)




#%%
##############
####   Create figures that represent conductivity and depolarization
####   versus the concentration of HCl
##############

with PdfPages('Figures/Depolarization_H2O_HCl.pdf') as pdf:
	plt.figure()
	plt.xlim([0, 1.05])
	plt.ylabel(r'$Depolarization$')
	plt.xlabel('HCl concentration [mol/L]')
	plt.ylim([0, 25])
	plt.plot(Depo[:,0], Depo[:,1],'.',color='r',label='')
	plt.errorbar(Depo[:,0], Depo[:,1], yerr=Depo[:,2],color='r',linestyle='None')
	pdf.savefig()
	plt.close()



conche, condur, condurerr = np.loadtxt('Conduction%s_H2O_HCl.dat' % date,comments='#',delimiter='\t',usecols=(0,1,2), unpack=True)

with PdfPages('Figures/Conductivity_H2O_HCl.pdf') as pdf:
	plt.figure()
	plt.xlim([0, 1.05])
	plt.ylabel(r'$\kappa$')
	plt.xlabel('HCl concentration [mol/L]')
	plt.ylim([0, 35])
	plt.plot(Condu[:,0], Condu[:,1],'.',color='r',label='')
	plt.errorbar(Condu[:,0], Condu[:,1], yerr=Condu[:,2],color='r',linestyle='None')
	pdf.savefig()
	plt.close()





#%%
##############
####   Shows the results of the fitting procedures
##############
show=int(input("Show figures (1/0): "))
if(show==1):
    n=5
    for par in Parameters:
        
        epsReal, epsImag = eps[:,1+2*n], eps[:,2+2*n]

        xN = x[112:]
        epsR = epsReal[112:]
        epsI = epsImag[112:]
        
        plt.figure()
        plt.xlim([1e9, 5e10])
        plt.xlabel('Frequency [Hz]')
        plt.ylim([0, 100])
        plt.xscale('log')
        plt.plot(xN, epsR,'.',label='')
        plt.plot(xN, ((epsI)*(-1.0)),'.',label='')
        plt.plot(xN, ((epsI)*(-1.0))-(FU2.Conduction(xN, par[6])),'.',label='')
#        plt.plot(xN, (FU2.Conduction(xN, par[6])),'.',label='')

        plt.title('Concentration: {} M'.format(par[0]))
    
        plt.plot(xN, FU2.Cole(xN, par[3], par[4]*1e-12, par[5]).real,label='')
        plt.plot(xN, (FU2.Cole(xN, par[3], par[4]*1e-12, par[5]).imag)*(-1),label='')
        plt.show()
        
        n+=1

