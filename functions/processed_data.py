# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:35:33 2019

@author: kasum
"""

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
    
    
##############################################################
### LOAD PROCESSED DATA    
##############################################################    
path= r'F:\EphysData\Experiments\190719_ks2\KA41-190719\KA41-190719'

episodes=pd.Series(['sleep','wake','wake','wake','wake','wake','wake'])
events= list(np.where(episodes == 'wake')[0].astype('str'))
spikes, shank                        = loadSpikeData(path)
#n_channels, fs, shank_to_channel   = loadXML(path)
position                            = loadPosition(path, events, episodes)
wake_ep                             = loadEpoch(path, 'wake', episodes)


sz=int(len(spikes.keys())/4)+1
for i in range(len(wake_ep)):
    figure();
    ep=nts.IntervalSet(start=wake_ep.loc[i,'start'], end=wake_ep.loc[i, 'start']+6e8)
    tc=computeAngularTuningCurves(spikes,position['ry'],ep,60)
    for x in spikes.keys():
        subplot(sz,4,x+1, projection='polar')
        plot(tc[x], label=str(x))
        legend()

#path_plot(wake_ep, position)         
            
            
#####################################################################################            

    
   

