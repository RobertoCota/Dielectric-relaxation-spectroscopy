import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

from scipy.integrate import quad
from scipy.integrate import romberg
from scipy.integrate import trapz

from scipy.optimize import curve_fit



def	Debye(x, S, tau):
	return 5.915+(S)/(1+(1j*(2*np.pi)*(tau)*(x)))

def	Cole(x, S, tau, alfa):
	return 5.915+(S)/(1+((1j*(2*np.pi)*(tau)*(x))**(1-alfa)))

def	ColeCon(x, S, tau, alfa, sigma):
	return 5.915+(S)/(1+((1j*(2*np.pi)*(tau)*(x))**(1-alfa)))-(sigma)/(2*np.pi*(8.8541878176e-12)*(x))*1j

def	ImagColeCon(x, S, tau, alfa, sigma):
	return ((S)*(((((2*np.pi)*(tau)*(x))**(1-alfa))*(np.cos(0.5*np.pi*alfa)))))/(1+(((2*np.pi)*(tau)*(x))**(2-2*alfa))+((2)*(((2*np.pi)*(tau)*(x))**(1-alfa))*(np.sin(0.5*np.pi*alfa))))+((sigma)/(2*np.pi*(8.8541878176e-12)*(x)))

def	Conduction(x, sigma):
	return (sigma)/(2*np.pi*(8.8541878176e-12)*(x))



def	SnH(cHCl, rhoSol, S0, DS0):

	rhoH20=997.00
	MHCl=36.46094
	C0=55.3197063825819
	MH2O=18.01528
	DrhoSol=0.005

	sn=(((rhoSol)*(S0))/(MH2O*C0))*(rhoH20/(rhoH20+(cHCl*MHCl)))

	Dsn=np.sqrt(((((1.0)*(S0))/(MH2O*C0))*(rhoH20/(rhoH20+cHCl*MHCl))**2)*((DrhoSol)**2)+((((rhoSol)*(1.0))/(MH2O*C0))*(rhoH20/(rhoH20+cHCl*MHCl))**2)*((DS0)**2))

	
	return [sn,Dsn]



def depolarizationparametrization(x,A,D):
	return A*x-D*np.power(x,3.0/2.0)

def Conductionparametrization(x,a,b):
	return a*x-b*np.power(x,3.0/2.0)





