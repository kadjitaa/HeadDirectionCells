#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:38:43 2019

@author: kasum
"""
#https://www.jneurosci.org/content/29/2/493
import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *
from pycircstat.tests import rayleigh
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from astropy.visualization import hist

import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
###############################################################
# PARAMETERS
###############################################################
data_directory=r'/Volumes/MyBook/EphysData/Experiments/200618/KA60-200618/KA60-200618'

episodes= ['wake','wake']#Modify this to suite the conditions you ave
events=['0','1'] #ids into the csvs in chro
n_analogin_channels = 2
channel_optitrack=1 #calls the second opened ch
spikes,shank= loadSpikeData(data_directory) #shank tells the number of cells on each shank
n_channels, fs, shank_to_channel = loadXML(data_directory)  #shank to channel 
position= loadPosition(data_directory,events,episodes,n_analogin_channels,channel_optitrack)
wake_ep=loadEpoch(data_directory,'wake',episodes)
sleep

###################################################################################
# ANALYSIS
###################################################################################
#Epochs

'''
wake_ep_1=nts.IntervalSet(start=wake_ep.loc[0,'start'], end =wake_ep.loc[0,'start']+6e8)
wake_ep_2=nts.IntervalSet(start=wake_ep.loc[3,'start'], end =wake_ep.loc[3,'start']+6e8)
wake_ep_3=nts.IntervalSet(start=wake_ep.loc[3,'start'], end =wake_ep.loc[3,'start']+3e+8)
#wake_ep_4=nts.IntervalSet(start=wake_ep.loc[3,'start'], end =wake_ep.loc[3,'start']+6e8)
#wake_ep_5=nts.IntervalSet(start=wake_ep.loc[4,'start'], end =wake_ep.loc[4,'end'])
#wake_ep_6=nts.IntervalSet(start=wake_ep.loc[5,'start'], end =wake_ep.loc[5,'end'])
#wake_ep_7=nts.IntervalSet(start=wake_ep.loc[6,'start'], end =wake_ep.loc[6,'end'])

#Tuning Curves
tuning_curves_1=computeAngularTuningCurves(spikes,position ['ry'],wake_ep_1,60)
tuning_curves_2=computeAngularTuningCurves(spikes,position ['ry'],wake_ep_2,60)
tuning_curves_3=computeAngularTuningCurves(spikes,position ['ry'],wake_ep_3,60)
#tuning_curves_4=computeAngularTuningCurves(spikes,position ['ry'],wake_ep_4,60)
#tuning_curves_5=computeAngularTuningCurves(spikes,position ['ry'],wake_ep_5,60)
#tuning_curves_6=computeAngularTuningCurves(spikes,position ['ry'],wake_ep_6,60)
#tuning_curves_7=computeAngularTuningCurves(spikes,position ['ry'],wake_ep_7,60)
'''
###############################################################
# PLOTTING
###############################################################


#from astropy.units import Quantity
#import astropy
#astropy.stats.circstats.vonmisesmle(array(position['ry'].restrict(wake_ep_2).realign(spikes[1].restrict(wake_ep_2))))


#Path and Polar Plots
#path_plot(wake_ep,position)

sz=(int(len(spikes.keys()))/4)+1
for i in range(len(wake_ep)):
    ep=nts.IntervalSet(start=wake_ep.loc[i,'start'], end=wake_ep.loc[i,'start']+6e8)
    tc=computeAngularTuningCurves(spikes,position['ry'],ep,60)
    figure()
    for x in spikes.keys():
        subplot(sz,4,1+x, projection='polar')
        plot(tc[x])

plt.suptitle('KA63-200703_90deg Floor Rotation')
  
sys.exit()

#HD Stats
stats=findHDCells(tuning_curves_2,wake_ep_2,spikes,position['ry'])
hd_cells=np.where(stats['hd_cells']==True)[0]

   
figure()
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tuning_curves_2[i],label=str(i),color='r', linewidth=2)
    ax2.set_xticklabels([])
    legend()






##############################################################################
gs = gridspec.GridSpec(2,2)

fig=figure(figsize=(3.9,1.96))
for i,x in enumerate(hd_cells):
    ax=subplot(gs[0,i], projection='polar') 
    

for x in spikes.keys():
    ax=subplot(1,2,x+1, projection='polar')
    plot(tcurv_dark[x],label=str(x),color='k', linewidth=2)
    if x==0:
        remove_polarAx(ax,False)
    else:
        remove_polarAx(ax,True)

fig=figure()       
for i,x in enumerate(spikes):
    ax2=subplot(gs[0,i], projection='polar') 
    plot(tuning_curves_1[x],label=str(i),color='k', linewidth=2)
    remove_polarAx(ax2,True)
    legend()
    
for i,x in enumerate(spikes):
    ax3=subplot(gs[1,i], projection='polar') 
    plot(tuning_curves_2[x],label=str(i),color='magenta', linewidth=2)
    remove_polarAx(ax3, True)
    legend()
cond='rd1-Standard Openfield_ '    
fig.suptitle( cond + data_directory.split("\\")[-1], size=18)   

patch1 = mpatches.Patch(color='k', label='Exposure 1')
patch2=mpatches.Patch(color='magenta', label='Exposure 2')
patch3=mpatches.Patch(color='green', label='cue @ 90deg')
plt.legend(handles=[patch1,patch2],loc='top right',bbox_to_anchor=(-1.5,2.2),fontsize=11)

data_files='/Users/Mac/HD_Drift/Figs/'
filename='rd1_OF12_KA54-200224.svg'

plt.savefig(data_files+filename,dpi=600, format='svg', bbox_inches="tight", pad_inches=0.05)


###############################################################
decoded_ang=np.unwrape()
true_ang=np.unwrap()



plot(time_indx*1000,unwrap_pos)
plot(position['ry'].restrict(wake_ep_1).index.values,true_ang)

idx=np.digitize(true_ang.index.values,bins) #the bins should be the bins of the decoded since it is the one with the lowest sample
down_true_ang=true_ang.groupby(idx).mean()
##########################################################

animal_id='KA30-190430_tcurves'
dir=data_directory+'/Analysis'

epochs=nts.IntervalSet(pd.read_hdf(dir+'/BehavEpochs.H5'))
position=pd.read_hdf(dir+'/'+'Position.H5')
position = nts.TsdFrame(t = position.index.values, d = position.values, columns = position.columns, time_units = 's')
spikes,shank= loadSpikeData(data_directory) #shank tells the number of cells on each shank

tcurv={}
for i in range(len(epochs)):
    tcurv[i]=computeAngularTuningCurves(spikes,position ['ry'],nts.IntervalSet(epochs.loc[i,'start'],epochs.loc[i,'end']),60)

np.save(os.path.join(dir, animal_id), tcurv)
######################




#ANALYSIS NOTES
"""examine HD tuning at different trajectories along the corners


"""












       


