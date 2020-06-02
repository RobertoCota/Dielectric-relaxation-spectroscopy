# -*- coding: utf-8 -*-
"""
Created on Thu Oct  19 11:26:11 2017

@author: Cota
"""

#Script for visualization of raw permittivities before fitting analyses

#To make this thing work you need the scat2perm file. 

from scat2perm import epsair, epsw, epsme, epset
#The scat2perm file that is loaded here contains thepermittivities 
#of well-defined samples.

import numpy as np
import matplotlib.pyplot as plt



#%%
#File that contains the names of the samples
namesData = 'Sample_Names_20171017.txt'

#Reading the names of the samples 
epsName = np.loadtxt(namesData,comments='#',dtype='str')

#Number of samples which epsilon has been calculated and stored from load_scat2perm_example (Check script!)  
Nsamples=len(epsName)


#%%
#File that contains all the experimentally calculated permittivities
epsData = 'Permittivity_Curves_20171017.dat'

#Reading the experimentally calculated permitivities
eps = np.loadtxt(epsData,comments='#',delimiter='\t')

#Defining the frequencies by array "x".
x=eps[:,0]

#%%
#Plotting the raw files by species
for i in range(int(Nsamples)):

      #Reading the raw data of epsilon
	epsreal=eps[:,1+2*i]
	epsimag=eps[:,2+2*i]
 

	plt.figure()
	plt.xscale('log')
 
	plt.xlim([1e8, 5e10])
	plt.ylim([0, 130])
 
	plt.title(epsName[i])
 
	plt.plot(x,epsreal,'.',label='Real')
	plt.plot(x,epsimag*(-1),'.',label='Imag')
 
	if(i==0):
		plt.plot(x,epsair(x).real,label= 'Ref: Re(Air)')
		plt.plot(x,(epsair(x).imag)*(-1.0),label= 'Ref: Im(Air)')

	if(i==1):
		plt.plot(x,epsw(x).real,label= 'Ref: Re(H2O)')
		plt.plot(x,(epsw(x).imag)*(-1.0),label= 'Ref: Im(H2O)')

	if(i==3):
		plt.plot(x,epsme(x).real,label= 'Ref: Re(Methanol)')
		plt.plot(x,(epsme(x).imag)*(-1.0),label= 'Ref: Im(Methanol)')

	if(i==4):
		plt.plot(x,epset(x).real,label= 'Ref: Re(Ethanol)')
		plt.plot(x,(epset(x).imag)*(-1.0),label= 'Ref: Im(Ethanol)')


	plt.legend()
	plt.show()
###################################
