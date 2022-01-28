# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:44:06 2019

@author: kasum
"""
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from pylab import *
import os, sys
from functions import *

############
# PARAMETERS
############
episodes= ['sleep','wake','wake','hf','hf','hf'] #Modify this to suite the conditions you ave
events=['2','3','4','5'] #ids into the csvs in chro
n_analogin_channels = 2
channel_optitrack=1 #calls the second opened ch

directory ='C:\Users\kasum\Documents\Files\2P\H52P\suite2p\plane'


spikes,shank= loadSpikeData(data_directory) #shank tells the number of cells on each shank
n_channels, fs, shank_to_channel = loadXML(data_directory)  #shank to channel 

wake_ep=loadEpoch(data_directory,'wake',episodes)
hf_ep=loadEpoch(data_directory,'hf',episodes) #calls the epoch for hf

#VisualStim (headfixed)  
name='/KA28-190405_4_analogin.dat' # directory to file with ttl of drifting condition
gratings_ep=makeHFEpochs(data_directory,name)

gratings_fr=pd.DataFrame(index=spikes.keys(),columns= gratings_ep.keys())

for c in gratings_ep.keys():
    for i in spikes:
        spk=spikes[i].restrict(gratings_ep[c])
        gratings_fr.loc[i,c]=float(len(spk))/len(gratings_ep[c]) 
        gratings_fr.loc[i,c]=gratings_fr.loc[i,c]/gratings_ep[c].tot_length('s')
 
    
grating_peakRate=pd.DataFrame (index=spikes.keys(),columns=pd.MultiIndex.from_product((gratings_ep.keys(),np.arange(10))))   
for c in gratings_ep.keys():
    for e in gratings_ep[c].index.values:
        for i in spikes:
            epi = nts.IntervalSet(gratings_ep[c].iloc[e,0], gratings_ep[c].iloc[e,1])
            spk = spikes[i].restrict(epi)
            grating_peakRate.loc[i,(c,e)] = float(len(spk))/epi.tot_length('s')

            
#gratings_ep['static_left'].as_units('s')# you can essetially change units here


#plot
gratings_fr.plot.bar(figsize=(15,10))
sys.exit()

title('HD Tuning to Drifting Grating',fontsize=20)
xlabel('ADn cell ID',fontsize=18)
ylabel('Mean Firing Rate (Hz)',fontsize=18)
xticks(fontsize=16)
yticks(fontsize=16)
legend(fontsize=16)
show()

plt.savefig('Grating tuning curve2.jpg', dpi=600, format='jpeg')






















