# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:28:57 2019

@author: kasum
"""
import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *
from sympy.combinatorics import Permutation

#############################################################
### CROSS CORROLOGRAMS FOR SELECTED CELL PAIRS###############
#############################################################
#cc_light=cc_light.to_hdf(data_files+'/cc.h5',mode='a',key='cc') #save file
cc=pd.read_hdf(data_files+'/cc.h5')
############################################################################
#PARAMETERS
############################################################################

###############################################################
def qwik_cc(cell_a,cell_b, spikes, ep):
    t1=spikes[cell_a].restrict(ep).as_units('ms')
    t1_t=t1.index.values
    
    t2=spikes[cell_b].restrict(ep).as_units('ms')
    t2_t=t2.index.values
    
    # Let's say you want to compute the autocorr with 10 ms bins
    binsize = 5
    # with 200 bins
    nbins = 400 #400
    
    autocorr_0 = crossCorr(t1_t, t2_t, binsize, nbins)
    
    # The corresponding times can be computed as follow 
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    
    # Let's make a time series
    autocorr_0 = pd.Series(index = times, data = autocorr_0)
    
    mean_fr_0 = len(t1)/ep.tot_length('s')
    autocorr_0 = autocorr_0 / mean_fr_0
    
    return autocorr_0
#autocorr_0.loc[0] = 0.0

###########################################################################
##PLOTS
###########################################################################

figs,ax=subplots()
c='darkgrey'
ax.plot(autocorr_0,c)
plt.fill_between(autocorr_0.index,autocorr_0.values,0, color=c)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#ax.spines['bottom'].set_position(('axes',0))
ax.set_ylim(0,autocorr_0.values.max())
ax.set_ylabel('Norm. correlation',size=21)
ax.set_xlabel('Time lag(ms)',size=21)
ax.xaxis.labelpad = 1 #sets the position of the xlabel

top=0.963,
bottom=0.162,
left=0.121,
right=0.975,
hspace=0.2,
wspace=0.2
sys.exit()

xticks = ax.xaxis.get_major_ticks()
tks_x=2,4,6,8
for i,x in enumerate(tks_x):
    xticks[x].label.set_visible(False) #removes the tick
    xticks[x].tick1line.set_visible(False)
 #removes the tick mark


ax.tick_params(labelsize=19)
#ax.set_xticklabels(fontsize=12)    
sys.exit()
yticks = ax.yaxis.get_major_ticks()    
tks_y=1,3,5,7
for i,x in enumerate(tks_y):
    yticks[x].label.set_visible(False) #removes the tick
    yticks[x].tick1line.set_visible(False) #removes the tick mark


xticks[3].label.set_visible(False)
xticks[3].tick1line.set_visible(False)


#############################################################
### CROSS CORROLOGRAMS FOR SELECTED CELL PAIRS###############
#############################################################
#Load data
data_files='C:/Users/kasum/Documents/HD_Drift/data'
dark_tc=pd.read_hdf(data_files+'\dark_tc.h5')

cc=compute_CrossCorrs()
#cc_dark = light.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 0.5)

#Post computing cross_corr

cell=pairs(tcurv_2) #tcurve is the input to this function





#Figures



fig,ax=subplots(figsize=(7,5))

subplot(121)
tmp=light[cell] #sorts the cross_corr mat based on the angular differences

tmp = tmp - tmp.mean(0)
tmp = tmp / tmp.std(0)
tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))

imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')

times = light.index.values
xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])], fontsize = 14)	
#yticks([0, len(cell)-1], [1, len(cell)], fontsize = 6)
gca().set_ylabel('Cell Pairs', fontsize=16)
title('Light')
xlabel("Time lag (ms)", fontsize = 16)
gca().tick_params(labelsize=14)


subplot(122)
tmp=dark[cell] #sorts the cross_corr mat based on the angular differences
tmp = tmp - tmp.mean(0)
tmp = tmp / tmp.std(0)
tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))


imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')

times = dark.index.values
xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])], fontsize = 14)	
yticks([])

title('Dark')
xlabel("Time lag (ms)", fontsize = 16)
gca().tick_params(labelsize=14)



















#needed if you want to quickly merge different epochs to construsct the cross_corrolograms
'''all_tc=tuning_curves_2
all_tc[all_tc.columns+shape(tuning_curves_2)[1]]=tuning_curves_2
'''

def pairs(tcurves):
    '''generates cell pairs based on angular difference in ascending order
    The input to the function is a dataframe of tuning curves'''
    
    pf=pd.DataFrame(index=[0],columns=tcurves.columns)
    for i in tcurves.columns:
        pf[i]=tcurves[i].idxmax()
        
    cells=list(combinations(tcurves.columns,2))
    ang_diff=pd.DataFrame(index=[0],columns=cells)
    for i,x in ang_diff.columns:
        unwrap_diff=np.unwrap(pf[i].values)-np.unwrap(pf[x].values)
        ang_diff[(i,x)]=abs(np.arctan2(np.sin(unwrap_diff),np.cos(unwrap_diff)))

    cell_pairs=ang_diff.T.sort_values(by=[0]).index
    return cell_pairs