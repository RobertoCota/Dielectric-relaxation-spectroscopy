# -*- coding: utf-8 -*-
"""
Created on Wed May  11 13:07:42 2016

@author: Cota
"""

import numpy as np
from scipy.optimize import fmin_powell

#These functions presented in this script are based on a script originally 
#written in MATLAB by Niklas Ottoson and Johannes Hunger. 

#The first translation into Python was made by Tibert van der Loop.


#######     PERMITTIVITY OF DIFFERENT SAMPLES     ########

#literature values interpolated for 23oC

def epsair(fr):
	return fr*0+1.+1j*0

#Permittivity of water at 23oC from https://doi.org/10.1021/jp982977k
def epsw(fr):
	return 5.915+(79.0877-5.915)/(1+(1j*(2*np.pi)*(8.70251e-12)*(fr)))

#Permittivity of a solution of 4M of NaCl in water at 23oC.
def epsnacl(fr):
	return (43.775-5.63)/(1+(1j*2*np.pi*fr*7.0525e-12)**(1.-.1865))+5.63 - 1j*22.833/2/np.pi/fr/8.854187817e-12

def epsme(fr):
	return (32.63-5.99)/(1+1.0*1j*fr*2*np.pi*51.6e-12)+(5.99-4.97)/(1+1.0*1j*fr*2*np.pi*7.74e-12)+(4.97-2.83)/(1+1.0*1j*fr*2*np.pi*1.16e-12)+2.83 # Methanol, Reference: Values: pickl_1998_diss, page 65

def epset(fr):
	return (24.32-4.49)/(1+1.0*1j*fr*2*np.pi*163e-12)+(4.49-3.82)/(1+1.0*1j*fr*2*np.pi*8.97e-12)+(3.82-2.69)/(1+1.0*1j*fr*2*np.pi*1.81e-12)+2.69 # Ethanol, Reference: Values: pickl_1998_diss, page 65

def epspc(fr):
	return (64.97-4.47)/(1+1.0*1j*fr*2*np.pi*43.e-12)**.91+(4.47-2.42)/(1+1.0*1j*fr*2*np.pi*0.57e-12)+2.42 # Propylene Carbonat, Reference: Values: BNPCBC, page 1226

def epsdma(fr):
	return (38.25-3.97)/(1+1.0*1j*fr*2*np.pi*15.8e-12)+(3.97-2.98)/(1+1.0*1j*fr*2*np.pi*0.95e-12)+2.98 # N,N-Dimethylacetamide, Reference: Values: dmadmfnmf, page 57




#%%
def geteps_cutoff(freq,S1,eps1,S2,eps2,S3,eps3,Smeas,l,p0=[40,-10]):
	"""
	Calculates permitivity from scattering parameters default 1=air 2=h2o 3=4M_NaCl
	This probe is for low frequency measurements.
 
	Literature reference: https://doi.org/10.1088/0957-0233/7/4/010
 
	"""
	c = 299792458
	Z1 = 1/(0.93213/50.*eps1**0.5*np.tanh(1j*2.*np.pi*freq*l/c*eps1**0.5))
	S1id = (Z1-50.)/(Z1+50.)
	Z2 = 1/(0.93213/50.*eps2**0.5*np.tanh(1j*2.*np.pi*freq*l/c*eps2**0.5))
	S2id = (Z2-50.)/(Z2+50.)
	Z3 = 1/(0.93213/50.*eps3**0.5*np.tanh(1j*2.*np.pi*freq*l/c*eps3**0.5))
	S3id = (Z3-50.)/(Z3+50.)
	Sas = S1id
	Sms = S1
	Sao = S2id
	Smo = S2
	Sal = S3id
	Sml = S3
	ed = (Sal*Sml*Smo*Sas-Sal*Sml*Sao*Sms-Sal*Sms*Smo*Sas-Sml*Sas*Sao*Smo+Sml*Sas*Sao*Sms+Sal*Sms*Sao*Smo)/(Sml*Sal*Sas-Sal*Sml*Sao-Sal*Sms*Sas-Smo*Sao*Sas+Sao*Sms*Sas+Sal*Sao*Smo)
	es = -(Smo*Sas-Sal*Smo-Sml*Sas+Sal*Sms-Sao*Sms+Sml*Sao)/(Sml*Sal*Sas-Sal*Sml*Sao-Sal*Sms*Sas-Smo*Sao*Sas+Sao*Sms*Sas+Sal*Sao*Smo)
	er = (-Sal*Sml*Smo**2*Sas**2+Sal**2*Sml*Smo**2*Sas+Sal*Sml**2*Smo*Sas**2+Sal**2*Sml*Sao*Sms**2-Sal*Sml*Sao**2*Sms**2+Sal*Sml**2*Sao**2*Sms+Sal*Sms*Smo**2*Sas**2-Sal**2*Sms*Smo**2*Sas+Sal**2*Sms**2*Smo*Sas+Sml*Sas**2*Sao*Smo**2-Sml**2*Sas**2*Sao*Smo+Sml**2*Sas*Sao**2*Smo+Sml**2*Sas**2*Sao*Sms+Sml*Sas*Sao**2*Sms**2-Sml**2*Sas*Sao**2*Sms+Sal**2*Sms*Sao*Smo**2-Sal**2*Sms**2*Sao*Smo+Sal*Sms**2*Sao**2*Smo-Smo*Sao**2*Sas*Sms**2-Sml*Sal**2*Sas*Sms**2+Sms*Smo**2*Sao**2*Sas+Sms*Sml**2*Sal**2*Sas-Sms*Sas**2*Smo**2*Sao-Sms**2*Sas**2*Smo*Sal+Sms**2*Sas**2*Smo*Sao-Sms**2*Sas**2*Sml*Sao-Sms*Sas**2*Sml**2*Sal+Sms**2*Sas**2*Sml*Sal-Sal**2*Sml**2*Smo*Sas-Sml*Sas*Sao**2*Smo**2-Sms*Sml**2*Sal**2*Sao-Sms*Smo**2*Sao**2*Sal+Sal**2*Sml**2*Smo*Sao-Sal**2*Sml*Smo**2*Sao-Sml**2*Sao**2*Smo*Sal+Sml*Sao**2*Smo**2*Sal)/((Sml*Sal*Sas-Sal*Sml*Sao-Sal*Sms*Sas-Smo*Sao*Sas+Sao*Sms*Sas+Sal*Sao*Smo)**2)
	Ssapcorr=(-Smeas+ed)/(-es*Smeas+es*ed-er)
 
	def Zsap(x):
		return 1./(0.93213/50.*(x[0]+1j*x[1])**0.5*np.tanh(1j*2*np.pi*freq*l*(1.0/c)*(x[0]+1j*x[1])**0.5))
  
	def fun(x):
		return (np.real((Zsap(x)-50.)/(Zsap(x)+50.))-np.real(Ssapcorr))**2+(np.imag((Zsap(x)-50.)/(Zsap(x)+50.))-np.imag(Ssapcorr))**2
  
	res = fmin_powell(fun,p0,xtol=0.5E-2,ftol=0.5E-2,maxfun=1000,maxiter=1000)
	return res[0]+res[1]*1j
	




#%%
def getepsopen185(freq,S1,eps1,S2,eps2,S3,Smeas,p0=[40,-10]):
	"""
	------------------------------------------
	getepsopen185(freq,S1,eps1,S2,eps2,S3,Smeas)
	------------------------------------------
	This probe is  high frequencies up to ~67GHz (VNA )
	gets permittivity spectra for 1.85mm open probe measurements
	freq:         frequencies [Hz]
	S1, S2, S3:   complex S11 for three calibration standards (S3 must be short)
	eps1, eps2:   complex permittivity of three calibration standards
	Smeas:        sample scattering parameter
  
	Literature reference: https://doi.org/10.1109/19.676718
	
	"""
	
	#I_n coefficients up to the 40th
	coeff=[0.00197411773207568,0,-4.19894358707932e-10,-2.88546656210754e-13,-1.29384352580990e-16,-4.58932868154231e-20,-1.37931546294612e-23,-3.63722494820569e-27,-8.59428044340735e-31,-1.84550093531113e-34,-3.63841916710768e-38,-6.63718500139070e-42,-1.12724676782898e-45,-1.79157203517365e-49,-2.67610576871519e-53,-3.77087478529098e-57,-5.02882914316356e-61,-6.36551654530715e-65,-7.66773913038527e-69,-8.81014702086766e-73,-9.67616370306041e-77,-1.01781624171493e-80,-1.02719719522406e-84,-9.96248183698378e-89,-9.29966876810441e-93,-8.36686102829917e-97,-7.26471816807600e-101,-6.09485825435239e-105,-4.94640695624807e-109,-3.88739656456150e-113,-2.96144378665187e-117,-2.18892418473362e-121,-1.57117216160598e-125,-1.09608498772632e-129,-7.43761599231371e-134,-4.91264867991582e-138,-3.16079685729866e-142,-1.98229040852451e-146,-1.21256282939175e-150,-7.23886651123377e-155]
	
	#Optimazed coefficients (alpha, beta, )
	cocorr =[4.110160e-001,4.079752e-001,-4.197636e-002,3.542403e-003,1.187986e-004 ]
  	
	#I_n^' Coefficients. Redefinition of the I_n coefficients
	coeff=coeff/((10**( cocorr[0] +cocorr[1]*(np.arange(40)) +cocorr[2]*(np.arange(40))**2  +cocorr[3]*(np.arange(40))**3  +cocorr[4]*(np.arange(40))**4 )))

	#Propagation constant within the dielectric material of the coaxial probe
	#See reference: Levine, Papas. "Theory of the circular diffraction antenna". 1951
	kc=freq*2*3.14159*(8.854187817e-12*4*np.pi*1e-7*4.1)**.5

	#Estimating S11 for air at the probe surface using literature values
	km=freq*2*3.14159*(8.854187817e-12*4*np.pi*1e-7*eps1)**.5
	Y=km**2/(np.pi*kc*np.log(.125/.023))*(1j*(coeff[0]-coeff[2]*km**2+coeff[4]*km**4-coeff[6]*km**6+coeff[8]*km**8-coeff[10]*km**10+coeff[12]*km**12-coeff[14]*km**14+coeff[16]*km**16-coeff[18]*km**18+coeff[20]*km**20-coeff[22]*km**22+coeff[24]*km**24-coeff[26]*km**26+coeff[28]*km**28-coeff[30]*km**30+coeff[32]*km**32-coeff[34]*km**34+coeff[36]*km**36-coeff[38]*km**38)+coeff[1]*km -coeff[3]*km**3+coeff[5]*km**5-coeff[7]*km**7+coeff[9]*km**9-coeff[11]*km**11+coeff[13]*km**13-coeff[15]*km**15+coeff[17]*km**17-coeff[19]*km**19+coeff[21]*km**21-coeff[23]*km**23+coeff[25]*km**25-coeff[27]*km**27+coeff[29]*km**29-coeff[31]*km**31+coeff[33]*km**33-coeff[35]*km**35+coeff[37]*km**37-coeff[39]*km**39)
	eSa=(1-Y)/(1+Y)

	#Estimating S11 for water at the probe surface using literature values
	km=freq*2*3.14159*(8.854187817e-12*4*np.pi*1e-7*eps2)**.5
	Y=km**2/(np.pi*kc*np.log(.125/.023))*(1j*(coeff[0]-coeff[2]*km**2+coeff[4]*km**4-coeff[6]*km**6+coeff[8]*km**8-coeff[10]*km**10+coeff[12]*km**12-coeff[14]*km**14+coeff[16]*km**16-coeff[18]*km**18+coeff[20]*km**20-coeff[22]*km**22+coeff[24]*km**24-coeff[26]*km**26+coeff[28]*km**28-coeff[30]*km**30+coeff[32]*km**32-coeff[34]*km**34+coeff[36]*km**36-coeff[38]*km**38)+coeff[1]*km -coeff[3]*km**3+coeff[5]*km**5-coeff[7]*km**7+coeff[9]*km**9-coeff[11]*km**11+coeff[13]*km**13-coeff[15]*km**15+coeff[17]*km**17-coeff[19]*km**19+coeff[21]*km**21-coeff[23]*km**23+coeff[25]*km**25-coeff[27]*km**27+coeff[29]*km**29-coeff[31]*km**31+coeff[33]*km**33-coeff[35]*km**35+coeff[37]*km**37-coeff[39]*km**39)
	eSw=(1-Y)/(1+Y)

	#Estimating S11 for the gold short at the probe surface using literature values
	km=freq*2*3.14159*(8.854187817e-12*4*np.pi*1e-7*1j*(-6e8/freq/8.854187817e-12))**.5; 
	# The conductivity is used here, because that dominates in the gold-short-circuit (a factor j*kappa/(omega*epsilonzero)), reference: Johannes' thesis, eq. 1.29
	Y=km**2/(np.pi*kc*np.log(.125/.023))*(1j*(coeff[0]-coeff[2]*km**2+coeff[4]*km**4-coeff[6]*km**6+coeff[8]*km**8-coeff[10]*km**10+coeff[12]*km**12-coeff[14]*km**14+coeff[16]*km**16-coeff[18]*km**18+coeff[20]*km**20-coeff[22]*km**22+coeff[24]*km**24-coeff[26]*km**26+coeff[28]*km**28-coeff[30]*km**30+coeff[32]*km**32-coeff[34]*km**34+coeff[36]*km**36-coeff[38]*km**38)+coeff[1]*km -coeff[3]*km**3+coeff[5]*km**5-coeff[7]*km**7+coeff[9]*km**9-coeff[11]*km**11+coeff[13]*km**13-coeff[15]*km**15+coeff[17]*km**17-coeff[19]*km**19+coeff[21]*km**21-coeff[23]*km**23+coeff[25]*km**25-coeff[27]*km**27+coeff[29]*km**29-coeff[31]*km**31+coeff[33]*km**33-coeff[35]*km**35+coeff[37]*km**37-coeff[39]*km**39)
	eSs=(1-Y)/(1+Y)

	#Calculating error terms
	Sas = eSs
	Sms = S3
	Sao = eSw
	Smo = S2
	Sal = eSa
	Sml = S1
	ed = (Sal*Sml*Smo*Sas-Sal*Sml*Sao*Sms-Sal*Sms*Smo*Sas-Sml*Sas*Sao*Smo+Sml*Sas*Sao*Sms+Sal*Sms*Sao*Smo)/(Sml*Sal*Sas-Sal*Sml*Sao-Sal*Sms*Sas-Smo*Sao*Sas+Sao*Sms*Sas+Sal*Sao*Smo)
	es = -(Smo*Sas-Sal*Smo-Sml*Sas+Sal*Sms-Sao*Sms+Sml*Sao)/(Sml*Sal*Sas-Sal*Sml*Sao-Sal*Sms*Sas-Smo*Sao*Sas+Sao*Sms*Sas+Sal*Sao*Smo)
	er = (-Sal*Sml*Smo**2*Sas**2+Sal**2*Sml*Smo**2*Sas+Sal*Sml**2*Smo*Sas**2+Sal**2*Sml*Sao*Sms**2-Sal*Sml*Sao**2*Sms**2+Sal*Sml**2*Sao**2*Sms+Sal*Sms*Smo**2*Sas**2-Sal**2*Sms*Smo**2*Sas+Sal**2*Sms**2*Smo*Sas+Sml*Sas**2*Sao*Smo**2-Sml**2*Sas**2*Sao*Smo+Sml**2*Sas*Sao**2*Smo+Sml**2*Sas**2*Sao*Sms+Sml*Sas*Sao**2*Sms**2-Sml**2*Sas*Sao**2*Sms+Sal**2*Sms*Sao*Smo**2-Sal**2*Sms**2*Sao*Smo+Sal*Sms**2*Sao**2*Smo-Smo*Sao**2*Sas*Sms**2-Sml*Sal**2*Sas*Sms**2+Sms*Smo**2*Sao**2*Sas+Sms*Sml**2*Sal**2*Sas-Sms*Sas**2*Smo**2*Sao-Sms**2*Sas**2*Smo*Sal+Sms**2*Sas**2*Smo*Sao-Sms**2*Sas**2*Sml*Sao-Sms*Sas**2*Sml**2*Sal+Sms**2*Sas**2*Sml*Sal-Sal**2*Sml**2*Smo*Sas-Sml*Sas*Sao**2*Smo**2-Sms*Sml**2*Sal**2*Sao-Sms*Smo**2*Sao**2*Sal+Sal**2*Sml**2*Smo*Sao-Sal**2*Sml*Smo**2*Sao-Sml**2*Sao**2*Smo*Sal+Sml*Sao**2*Smo**2*Sal)/((Sml*Sal*Sas-Sal*Sml*Sao-Sal*Sms*Sas-Smo*Sao*Sas+Sao*Sms*Sas+Sal*Sao*Smo)**2)
	Ssapcorr=(-Smeas+ed)/(-es*Smeas+es*ed-er)

	#Get permittivity of sample
	kc=freq*2*3.14159*(8.854187817e-12*4*np.pi*1e-7*4.1)**(.5)
 
	def fun(x):
		km=	freq*2*3.14159*(8.854187817e-12*4*np.pi*1e-7*(x[0]+1j*x[1]))**(.5)
		Y= km**2/(np.pi*kc*np.log(.125/.023))*(1j*(coeff[0]-coeff[2]*km**2+coeff[4]*km**4-coeff[6]*km**6+coeff[8]*km**8-coeff[10]*km**10+coeff[12]*km**12-coeff[14]*km**14+coeff[16]*km**16-coeff[18]*km**18+coeff[20]*km**20-coeff[22]*km**22+coeff[24]*km**24-coeff[26]*km**26+coeff[28]*km**28-coeff[30]*km**30+coeff[32]*km**32-coeff[34]*km**34+coeff[36]*km**36-coeff[38]*km**38)+coeff[1]*km -coeff[3]*km**3+coeff[5]*km**5-coeff[7]*km**7+coeff[9]*km**9-coeff[11]*km**11+coeff[13]*km**13-coeff[15]*km**15+coeff[17]*km**17-coeff[19]*km**19+coeff[21]*km**21-coeff[23]*km**23+coeff[25]*km**25-coeff[27]*km**27+coeff[29]*km**29-coeff[31]*km**31+coeff[33]*km**33-coeff[35]*km**35+coeff[37]*km**37-coeff[39]*km**39)
		SSS= (1-Y)/(1+Y)    
		return	(np.real(SSS)-np.real(Ssapcorr))**2+(np.imag(SSS)-np.imag(Ssapcorr))**2
  
	res = fmin_powell(fun,p0,xtol=1E-3,ftol=1E-3,maxfun=300,maxiter=300)
	return res[0]+res[1]*1j

