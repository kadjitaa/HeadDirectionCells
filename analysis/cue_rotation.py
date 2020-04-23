# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:26:29 2020

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
import math
###############################################################################
###Setting Directory and Params
###############################################################################
data_directory   = r'C:\Users\kasum\Desktop'
info             = pd.read_excel(os.path.join(data_directory,'experiments.xlsx')) #directory to file with all exp data info

strain='rd1' #you can equally specify the mouse you want to look at
exp='standard'
cond1='EnvA'
cond2='EnvB'
#cond3= info.cue_ang==90


#################################################################################
###Preselect Rows of Interest for group analysis
#################################################################################

idx=[] #index for all the rows that meet exp condition
idx2=[] #index for all the rows that meet cond1 and 2
for i in range(len(info)):
    if np.any(info.iloc[i,:].str.contains(exp)) and np.any(info.iloc[i,:].str.contains(strain)) : #and np.any(info.iloc[i,:].str.contains(cond1)):
        idx.append(i)
        if np.any(info.iloc[i,:].str.contains(cond1)==True) and np.any(info.iloc[i,:].str.contains(cond2)) and np.any(info.floor_ang[i]==90):
            idx2.append(i)

##############################################################################
###Within Animal Analysis
################################################################################
###Combined Datasets      
all_peaks=pd.DataFrame(columns=([cond1,cond2]))
all_means=pd.DataFrame(columns=([cond1,cond2]))
all_pfd=pd.DataFrame(columns=([cond1,cond2]))
all_vLength=pd.DataFrame(columns=([cond1,cond2]))
#all_stability=pd.DataFrame(columns=([cond1,cond2]))
all_circVar=pd.DataFrame(columns=([cond1,cond2]))
all_circMean=pd.DataFrame(columns=([cond1,cond2]))
all_info=pd.DataFrame(columns=([cond1,cond2]))
#all_tcurv1=[]


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
    #all_tcurv1=all_tcurv1.append(tcurv_1)

    ############################################################################################### 
    # FIRING RATE ANALYSIS
    ###############################################################################################
    stats=findHDCells(tcurv_1,ep1,spikes,position['ry'])
    hd=stats['hd_cells']==True
    mean_frate,peak_frate, pfd  = computeFiringRates(spikes,[ep1, ep2],[cond1,cond2],[tcurv_1,tcurv_2]) 
    #info_hd                         = computeInfo([ep1,ep2] ,spikes,position,[cond1,cond2])
    #mean_vLength                    = computeVectorLength(spikes,[ep1,ep2], position['ry'],[cond1,cond2])
    #spatial_corr                    = computeStability([ep1,ep2],spikes,position['ry'],[cond1,cond2])
    circ_mean,circ_var              = computeCircularStats([ep1,ep2],spikes,position['ry'],[cond1,cond2])
    
    ###############################################################################################
    ## MERGE ACROSS SESSIONS
    ###############################################################################################
    all_pfd=all_pfd.append(pfd[hd])
    all_means=all_means.append(mean_frate[hd])
    all_peaks=all_peaks.append(peak_frate[hd])
    #all_stability=all_stability.append(spatial_corr[hd])    
    #all_vLength=all_vLength,append(mean_vLength[hd])
    all_circVar=all_circVar.append(circ_var[hd])
    all_circMean=all_circMean.append(circ_mean[hd])
    #all_info=all_info.append(info_hd[hd])
    #all_tcurv1=all_tcurv1.append(tcurv_1)
    #all_tcurv2=all_tcurv2.append(tcurv_2)
    
    
gnat_180_wall =pd.DataFrame({'pfd':[all_pfd],'mfrates':[all_means],'pfrates':[all_peaks],
                          'circMean':[all_circMean], 'circVar':[all_circVar]})

data_files='C:/Users/kasum/Documents/HD_Drift/data'
gnat_180_wall.to_hdf(data_files+'/gnat_180_wall.h5',mode='a',key='gnat_180_wall') #save file

os.sys.exit()  
    
wt=pd.read_hdf(data_files+'\gnat_180_cue.h5' )
    
##################################################
###Ploting
##################################################    
figure()
for i in spikes.keys():
    rws=int(len(spikes.keys())/4)+1
    ax=subplot(rws,4,i+1, projection='polar')
    plot(tcurv_1[i])    
    
                
cueA=np.unwrap(all_pfd['floorA'].values *( 180/np.pi))
cueB=np.unwrap(all_pfd['floorB'].values *( 180/np.pi))
shift_diff=90-(abs(cueA-cueB))

fig=figure()
scatter(np.ones(len(shift_diff))*1.5,shift_diff)    
gca().set_ylim(-90,180)
gca().set_xticks([1])
gca().set_xticklabels(['180deg'])        

figure(); scatter(cueA,cueB)  
'''note
 Because HD cells tend to shift their preferred firing directions in
324 register with each other (e.g. Taube et al. 1990b, Knierim et al. 1995, Yoganarasimha et al.
325 2006), if multiple HD cells were recorded during the same session, PFD shifts between different
326 planes of locomotion were calculated as the average of the PFD shifts across all included HD
327 cells, such that each session had a single PFD shift
 -https://www-jneurosci-org.proxy3.library.mcgill.ca/content/jneuro/early/2020/02/27/JNEUROSCI.2789-19.2020.full.pdf
 increase the size of the scatter and compress the y_axis
 
 
 --find a way of doing a proper OSN lesion or inactivation besides the znc sulphate bcos it could be affectin vestibular sys
 '''
 
 