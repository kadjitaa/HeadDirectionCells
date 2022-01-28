# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:32:09 2020

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
data_directoryy   = r'C:\Users\kasum\Desktop'
info             = pd.read_excel(os.path.join(data_directoryy,'experiments.xlsx')) #directory to file with all exp data info

strain='wt' #you can equally specify the mouse you want to look at
exp='standard'
cond1='light'
cond2='dark'


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
all_light_autocorr=[];
all_dark_autocorr=[]; 
all_width=pd.DataFrame(columns=[0])
all_light_tc=[]
all_dark_tc=[]

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
    
    spikess, shank                       = loadSpikeData(path)
    #n_channels, fs, shank_to_channel   = loadXML(path)
    positions                            = loadPosition(path, events, episodes)
    wake_ep                             = loadEpoch(path, 'wake', episodes)
    
    ep1=nts.IntervalSet(start=wake_ep.loc[int(events[0])-1,'start'], end =wake_ep.loc[int(events[0])-1,'start']+6e+8)
    ep2=nts.IntervalSet(start=wake_ep.loc[int(events[-1])-1,'start'], end =wake_ep.loc[int(events[-1])-1,'start']+6e+8)
        
    tcurv_1 = computeAngularTuningCurves(spikess,positions['ry'],ep1,60)
    tcurv_2 = computeAngularTuningCurves(spikess,positions['ry'],ep2,60)
    

    stats=findHDCells(tcurv_2,ep2,spikes,position['ry'])
    hd=stats['hd_cells']==True

    #dark= compute_CrossCorrs(spikes,ep1, 10,1000)
    #light= compute_CrossCorrs(spikes,ep2, 10,1000)
    
    
    #dark_cc.append(dark)
    #light_cc.append(light)
    mean_frate,peak_frate, pfd      = computeFiringRates(spikes,[ep1, ep2],[cond1,cond2],[tcurv_1,tcurv_2]) 
    info_hd                         = computeInfo([ep1,ep2] ,spikes,position,[cond1,cond2])
    mean_vLength                    = computeVectorLength(spikes,[ep1,ep2], position['ry'],[cond1,cond2])
    spatial_corr                    = computeStability([ep1,ep2],spikes,position['ry'],[cond1,cond2])
    circ_mean,circ_var              = computeCircularStats([ep1,ep2],spikes,position['ry'],[cond1,cond2])
    #tcurv1_width                     = tc_width(tcurv_1,spikes)
    tcurv2_width                     = tc_width(tcurv_2,spikes)
    light_autocorr,_                = compute_AutoCorrs(spikes,ep2)
    dark_autocorr,_                 = compute_AutoCorrs(spikes,ep1)

    
    ###############################################################################################
    ## MERGE ACROSS SESSIONS
    ###############################################################################################
    
    all_pfd=all_pfd.append(pfd[hd])
    all_means=all_means.append(mean_frate[hd])
    all_peaks=all_peaks.append(peak_frate[hd])
    all_stability=all_stability.append(spatial_corr[hd])    
    all_vLength=all_vLength.append(mean_vLength[hd])
    all_circVar=all_circVar.append(circ_var[hd])
    all_circMean=all_circMean.append(circ_mean[hd])
    all_info=all_info.append(info_hd[hd])
    #all_tcurv1=all_tcurv1.append(tcurv_1)
    #all_tcurv2=all_tcurv2.append(tcurv_2)   
    all_light_tc.append(tcurv_2)
    all_dark_tc.append(tcurv_1)    
    all_width=all_width[0].append(tcurv2_width[hd])
    all_dark_autocorr.append(dark_autocorr[dark_autocorr.columns[hd]])
    all_light_autocorr.append(light_autocorr[light_autocorr.columns[hd]])
    

#dc=pd.concat([dark_cc[0],dark_cc[1]], axis=1,ignore_index=False )

#len(dc.columns)
#len(list(combinations(range(13),2)))
    
    
    

#all_dark_tc=pd.concat([all_dark_tc[0],all_dark_tc[1]], axis=1,ignore_index=True )


#all_light_tc=pd.concat([all_light_tc[0],all_light_tc[1]], axis=1, ignore_index=True) 

dark_light=pd.DataFrame([{'light_cc':light_cc,'light_ac':all_light_autocorr,'light_tc':all_light_tc,'dark_cc':dark_cc,'dark_ac':all_dark_autocorr,
            'dark_tc':all_dark_tc,'pfd':all_pfd,'mfrates':all_means,'pfrates':all_peaks,
            'circMean':all_circMean, 'circVar':all_circVar,'light_width':all_width,'vlength': all_vLength,
            'stability':all_stability, 'info':all_info,'drift':drift}])

dark_light.to_hdf(data_files+'/dark_light_dataset.h5',mode='a',key='dark_light_dataset')


a=pd.DataFrame({'light_width': all_width})
pd.DataFrame([a])
###################
## SAVING FILES
###################
data_files='C:/Users/kasum/Documents/HD_Drift/data'
dark_light.to_hdf(data_files+'/dark_light.h5',mode='a',key='dark_light')

drift=pd.read_hdf(data_files+'/drift_dataset.h5')
dark_cc=pd.read_hdf(data_files+'/dark_cc.h5')

#all_dark_tc.to_hdf(data_files+'/dark_tc.h5',mode='a',key='dark_tc')
#all_light_tc.to_hdf(data_files+'/light_tc.h5',mode='a',key='light_tc')


