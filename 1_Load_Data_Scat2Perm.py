# -*- coding: utf-8 -*-
"""
Created on Thu Oct  19 11:32:22 2017

@author: Cota
"""

#Script for caclulating permittivities from scattering 
#parameters from a DRS measurement using the VNA approach.

#To make this thing work you need the scat2perm file. 

import scat2perm as S2P
#The scat2perm file that is loaded here contains the functions 
#that are used to find the permittivities of the scattering 
#parameters using the three callibration standards.


import numpy as np
import matplotlib.pyplot as plt



#%%
###########     LIST AND CLASSIFY THE EXPERIMENTAL SAMPLES      ############

date = '20171017'

cd = 'Data/'

files = [
        ['20171017_Air_2.dat', '20171017_Air_3.dat'],
        ['20171017_H2O_1.dat', '20171017_H2O_2.dat', '20171017_H2O_3.dat'],
        ['20171017_Short_1.dat', '20171017_Short_2.dat', '20171017_Short_3.dat'],

        ['20171017_Methanol_1.dat', '20171017_Methanol_2.dat', '20171017_Methanol_3.dat'],
        ['20171017_Ethanol_1.dat', '20171017_Ethanol_2.dat', '20171017_Ethanol_3.dat'],


        ['20171017_Z_H2O_HCl_010M_S1_1.dat', '20171017_Z_H2O_HCl_010M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_010M_S2_1.dat', '20171017_Z_H2O_HCl_010M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_020M_S1_1.dat', '20171017_Z_H2O_HCl_020M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_020M_S2_1.dat', '20171017_Z_H2O_HCl_020M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_030M_S1_1.dat', '20171017_Z_H2O_HCl_030M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_030M_S2_1.dat', '20171017_Z_H2O_HCl_030M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_040M_S1_1.dat', '20171017_Z_H2O_HCl_040M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_040M_S2_1.dat', '20171017_Z_H2O_HCl_040M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_050M_S1_1.dat', '20171017_Z_H2O_HCl_050M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_050M_S2_1.dat', '20171017_Z_H2O_HCl_050M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_060M_S1_1.dat', '20171017_Z_H2O_HCl_060M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_060M_S2_1.dat', '20171017_Z_H2O_HCl_060M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_070M_S1_1.dat', '20171017_Z_H2O_HCl_070M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_070M_S2_1.dat', '20171017_Z_H2O_HCl_070M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_080M_S1_1.dat', '20171017_Z_H2O_HCl_080M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_080M_S2_1.dat', '20171017_Z_H2O_HCl_080M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_090M_S1_1.dat', '20171017_Z_H2O_HCl_090M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_090M_S2_1.dat', '20171017_Z_H2O_HCl_090M_S2_2.dat'],

        ['20171017_Z_H2O_HCl_100M_S1_1.dat', '20171017_Z_H2O_HCl_100M_S1_2.dat'],
        ['20171017_Z_H2O_HCl_100M_S2_1.dat', '20171017_Z_H2O_HCl_100M_S2_2.dat']
        ]
    
names = []
for file in files:
    names.append(file[0][9:][:-6])

#Save the name of the samples in a file
np.savetxt('Sample_Names_%s.txt' % date, np.array(names),fmt="%s")


#%%
###########     CHECK REPRODUCIBILITY      ############

#Check reproducibilty by doing multiple measurements. 
#If all goes well the data will match very well. 

#Plotting the raw files here serves as a check. 
#Irreproducibility may come from bubbles that stick on the 
#surface of the sample cell or temperature variations. 



#Defining the frequencies by array "t".
t = np.loadtxt(cd + files[0][0],comments='%',delimiter=',',usecols=(0,))

#Plotting the raw files by species
for i in range(len(files)):
	plt.figure()
	for j in range(len(files[i])):
		plt.plot(t,np.loadtxt(cd + files[i][j], comments='%', delimiter=',',usecols=(1,)), label='Sample {}'.format(j+1))
	plt.title(names[i])
	plt.legend(loc=2)
	plt.show()
#	a=input('Press Enter to continue!')

#Comment or uncomment the stop line accordingly if you have checked 
#the raw measurements and discarded the wrong data. 

#Delete the wrong files from the files array.

#print (stop)




#%%
###########     CALCULATE PERMITTIVITIES      ############

#Defining the frequencies by array "t".
t = np.loadtxt(cd + files[0][0] ,comments='%', delimiter=',', usecols=(0,))


#Read data files and average over same type of measurements.

S_data= np.zeros([301,len(files)],dtype=complex)
for i,f in enumerate(files):
	print (len(f))
	for fn in f:
		tmp=np.loadtxt(cd + fn,comments='%',delimiter=',',usecols=(1,2))
		S_data[:,i]+=tmp[:,0]+tmp[:,1]*1j
	S_data[:,i]/=len(f)




#Calculate permittivities from imported scattering data. 

#p0 is the initial guess for the lowest frequency. 
#After calculating the first permittivity (lowest frequency), 
#p0 takes the value of the result of the previous immediate iteration.
p0=[40,-10]

eps_data=np.zeros([301,len(files)],dtype=complex)


for j in range(eps_data.shape[1]):
	for i in range(eps_data.shape[0]):
		eps_data[i,j]=S2P.getepsopen185(t[i],S_data[i,0],S2P.epsair(t[i]),S_data[i,1],S2P.epsw(t[i]),S_data[i,2],S_data[i,j],p0=p0)
		p0=[eps_data[i,0].real,eps_data[i,0].imag]
		print (i,j)


print ('\n Analysis: Done! \n')




#%% 
###########     EXPORT CALCULATED PERMITTIVITIES      ############

#Write complex permittivities to file. 
#The numpy array containing complex numbers to a file. Therefore I defined the following function.

#The file can be opened with functions like 'loadtxt(filename dtype=comlpex)'
#for further analysis or data visualization

with open('Permittivity_Curves_%s.dat' % date,'w') as fp:
	fp.write('#%s\n#f[Hz]\n'% (date))
	#np.savetxt(fp, files[0], fmt='%.18e')
	for i in range(eps_data.shape[0]):
		fp.write('%s\t' % t[i])
		#fp.write('\t'.join([ "{0.real:.8e}{0.imag:+.8e}j".format(v) for v in eps_data[i] ]))
		fp.write('\t'.join([ "{0.real:.8e}\t{0.imag:+.8e}".format(v) for v in eps_data[i] ]))
		fp.write('\n')

