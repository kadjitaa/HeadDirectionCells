# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:06:39 2020

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
"""
data_directory   = r'C:\Users\kasum\Desktop'
info             = pd.read_excel(os.path.join(data_directory,'experiments.xlsx')) #directory to file with all exp data info

strain='rd1' #you can equally specify the mouse you want to look at
exp='standard'
cond1='EnvA'
cond2='EnvA'
#cond3= info.floor_ang==90


#################################################################################
###Preselect Rows of Interest for group analysis
#################################################################################

idx=[] #index for all the rows that meet exp condition
idx2=[] #index for all the rows that meet cond1 and 2
for i in range(len(info)):
    if np.any(info.iloc[i,:].str.contains(exp)) and np.any(info.iloc[i,:].str.contains(strain)) : #and np.any(info.iloc[i,:].str.contains(cond1)):
        idx.append(i)
        if np.any(info.iloc[i,:].str.contains(cond1)==True) or np.any(info.iloc[i,:].str.contains(cond2)):
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

all_light_tc=[]
all_dark_tc=[]
"""
strain='gnat'
#idx2=[3,16,17] #wt
#idx2=[55,62] #rd
idx2=[23,27,34,40]


#wt_envA=pd.DataFrame(columns=['EnvA'])
#rd_envA=pd.DataFrame(columns=['EnvA'])
#gnat_envA=pd.DataFrame(columns=['EnvA'])



gnat=pd.DataFrame()
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
    #sleep_ep                            =loadEpoch(path,'sleep',episodes)
    
    ep1=nts.IntervalSet(start=wake_ep.loc[int(events[0])-1,'start'], end =wake_ep.loc[int(events[0])-1,'start']+6e+8)
    ep2=nts.IntervalSet(start=wake_ep.loc[int(events[-1])-1,'start'], end =wake_ep.loc[int(events[-1])-1,'start']+6e+8)
        
    tcurv_1 = computeAngularTuningCurves(spikes,position['ry'],ep1,60)
    tcurv_2 = computeAngularTuningCurves(spikes,position['ry'],ep2,60)
    
    
    stats=findHDCells(tcurv_2,ep2,spikes,position['ry'])
    hd_cells=stats['hd_cells']==True
    
    infs=computePlaceInfo(spikes,position,ep2)
    infs=infs[hd_cells]
    gnat=gnat.append(infs)
    
    
    

    
    sw=slidingWinEp(ep2,diff(ep2)//2)
    

    sw_ep1=sw.loc[0]; sw_ep1=nts.IntervalSet(sw_ep1.start,sw_ep1.end)
    sw_ep2=sw.loc[1]; sw_ep2=nts.IntervalSet(sw_ep2.start,sw_ep2.end)
    #tmp,_=frate_maps(spikes,position,sw_ep1)
    #tmp1,_=frate_maps(spikes,position,sw_ep2)
    #sns.jointplot(position['x'].restrict(ep2),position['z'], kind='hex');legend([str(s)])
    figure();plot(position['x'].restrict(ep2),position['z'].restrict(ep2), label=str(s)); legend()
    
    
    
    tmp=all_frate_maps(spikes,position,sw_ep1)
    tmp1=all_frate_maps(spikes,position,sw_ep2)
    if strain=='wt':
        pearson_c=pd.DataFrame(index=spikes.keys(),columns=['EnvA'])
        for i in range(len(tmp)):
            pearson_c.loc[i]=scipy.stats.pearsonr(tmp[i].flatten(),tmp1[i].flatten())[0]        
        wt_envA=wt_envA.append(pearson_c[hd_cells])
        
    elif strain=='gnat':
        pearson_c=pd.DataFrame(index=spikes.keys(),columns=['EnvA'])
        for i in range(len(tmp)):
            pearson_c.loc[i]=scipy.stats.pearsonr(tmp[i].flatten(),tmp1[i].flatten())[0]
        gnat_envA=gnat_envA.append(pearson_c[hd_cells])
        
    elif strain=='rd1':
        pearson_c=pd.DataFrame(index=spikes.keys(),columns=['EnvA'])
        for i in range(len(tmp)):
            pearson_c.loc[i]=scipy.stats.pearsonr(tmp[i].flatten(),tmp1[i].flatten())[0]
        rd_envA=rd_envA.append(pearson_c[hd_cells])
    
    
    
    
    
    if strain=='wt':
        info=pd.DataFrame(index=spikes.keys(),columns=['inf'])
        for i in spikes.keys():
            pearson_c.loc[i]=scipy.stats.pearsonr(tmp[i].flatten(),tmp1[i].flatten())[0]        
        wt_envA=wt_envA.append(pearson_c[hd_cells])
        
    elif strain=='gnat':
        pearson_c=pd.DataFrame(index=spikes.keys(),columns=['EnvA'])
        for i in range(len(tmp)):
            pearson_c.loc[i]=scipy.stats.pearsonr(tmp[i].flatten(),tmp1[i].flatten())[0]
        gnat_envA=gnat_envA.append(pearson_c[hd_cells])
        
    elif strain=='rd1':
        pearson_c=pd.DataFrame(index=spikes.keys(),columns=['EnvA'])
        for i in range(len(tmp)):
            pearson_c.loc[i]=scipy.stats.pearsonr(tmp[i].flatten(),tmp1[i].flatten())[0]
        rd_envA=rd_envA.append(pearson_c[hd_cells])
        
        
        
        

    
    
grp_spatialCorrs=pd.DataFrame([{'wt_corr': wt_envA,'wt_inf':wt,'rd_corr':rd_envA,'rd_inf':rd,'gnat_cor':gnat_envA,'gnat_info':gnat}])
data_files='C:/Users/kasum/Documents/HD_Drift/data'
grp_spatialCorrs.to_hdf(data_files+'/grp_CorrsSpatial.h5',mode='a',key='grp_CorrsSpatial')


    
boxplot([wt_envA.EnvA,gnat_envA.EnvA,rd_envA.EnvA])  
figure(); boxplot([wt.values,gnat.values,rd.values])  
    
    
#corr_envA['blind']=gcorr_envA.values

gca().set_xticklabels(['wt','gnat','rd1'])
gca().set_ylabel('Spatial Correlation (r)')
scipy.stats.mannwhitneyu(gnat_envA.EnvA,rd_envA.EnvA)

