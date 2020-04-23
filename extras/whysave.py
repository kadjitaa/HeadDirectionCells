# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:41:20 2020

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

###############################################################################
###Setting Directory and Params
###############################################################################
data_directory   = r'C:\Users\kasum\Desktop'
info             = pd.read_excel(os.path.join(data_directory,'experiments.xlsx')) #directory to file with all exp data info

strain='wt' #you can equally specify the mouse you want to look at
exp='standard'
cond2='light'
cond1='dark'


#################################################################################
###Preselect Rows of Interest for group analysis
#################################################################################

idx=[] #index for all the rows that meet exp condition
idx2=[] #index for all the rows that meet cond1 and 2
for i in range(len(info)):
    if np.any(info.iloc[i,:].str.contains(exp)) and np.any(info.iloc[i,:].str.contains(strain)) : #and np.any(info.iloc[i,:].str.contains(cond1)):
        idx.append(i)
        if np.any(info.iloc[i,:].str.contains(cond1)==True) and np.any(info.iloc[i,:].str.contains(cond2)):
            idx2.append(i)

##############################################################################
###Within Animal Analysis
################################################################################
###Combined Datasets      
all_means=pd.DataFrame(columns=([cond1,cond2]))
all_peaks=pd.DataFrame(columns=([cond1,cond2]))
all_pfd=pd.DataFrame(columns=([cond1,cond2]))
all_info=pd.DataFrame(columns=([cond1,cond2]))
all_vLength=pd.DataFrame(columns=([cond1,cond2]))
all_stability=pd.DataFrame(columns=([cond1,cond2]))
all_circMean=pd.DataFrame(columns=([cond1,cond2]))
all_circVar=pd.DataFrame(columns=([cond1,cond2]))
all_light_tc=[];
all_dark_tc=[]; 


###############################################################################
###Data Processing
##############################################################################
for x,s in enumerate(idx2):
    path=info.dir[s].replace('\\',"/")
  
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    episodes = info.filter(like='T').loc[s]
    events  = list(np.where((episodes == cond1) | (episodes== cond2))[0].astype('str'))
    
    spikes, shank                       = loadSpikeData(path)
    #n_channels, fs, shank_to_channel   = loadXML(path)
    position                            = loadPosition(path, events, episodes)
    wake_ep                             = loadEpoch(path, 'wake', episodes)
    
    ep1=nts.IntervalSet(start=wake_ep.loc[int(events[0])-1,'start'], end =wake_ep.loc[int(events[0])-1,'end'])
    ep2=nts.IntervalSet(start=wake_ep.loc[int(events[-1])-1,'start'], end =wake_ep.loc[int(events[-1])-1,'end'])
        
    tcurv_1 = computeAngularTuningCurves(spikes,position['ry'],ep1,60)
    tcurv_2 = computeAngularTuningCurves(spikes,position['ry'],ep2,60)
    

    stats=findHDCells(tcurv_2,ep2,spikes,position['ry'])
    hd=stats['hd_cells']==True
    
    