# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:57:37 2019

@author: kasum
"""

import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *
import seaborn as sns
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

###############################################################################################
#DEFINE PARAMS
###############################################################################################

ep=wake_ep_1
cells=[0] #define the cell you want to look at
###############################################################################################

def f_ang_tcurves(ep, spikes, position):
    ang_ep=full_ang(ep,position['ry'])
    eps=np.arange(len(ang_ep))
    pairs=[1,2,3,4,21]
    cells_i=[1,2]
    for q in cells_i:
        cells=[q]
        fig=plt.figure()
        #fig.suptitle('#Cell_' +str(q),fontsize=25)
        for j,q in enumerate(pairs): #epoch of interest 
            f_ang_ep1=nts.IntervalSet(start=ang_ep.loc[j,'start'],end=ang_ep.loc[j,'end'])
            tcuve_f1=computeAngularTuningCurves(spikes,position['ry'],f_ang_ep1)   
            
            
            
            
            for i,x in enumerate(cells):
                ax=subplot(1,5,j+1, projection='polar')
                plt.plot(tcuve_f1[x],color='grey',linewidth=3)
                xticks = ax.xaxis.get_major_ticks()
                xticks[1].label.set_visible(False)
                xticks[3].label.set_visible(False)
                xticks[5].label.set_visible(False)
                xticks[7].label.set_visible(False)
                xticks = ax.xaxis.get_major_ticks()
                ax.set_yticks([])

                tck=[1,3,5,7]
                for i in tck:
                    xticks[i].set_visible(False)
                #ax.fill_between(tcuve_f1[x].index,tcuve_f1[x].values,0, color='darkgrey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                #legend()




alltc=f_ang_tcurves(ep1, spikes,position)
      
    az=subplot(5,5,18,projection='polar')
    plt.plot(tuning_curves_2[7],color='k',zorder=2)
    fill_between(tuning_curves_2[7].index,tuning_curves_2[7].values,0, color='red',zorder=2)
fig.suptitle('#KA41(wildtype)_cell 19', fontsize=25)
#ax.grid(False)             
az.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])

figure()
ax=subplot(351,projection='polar')
plot(tcurv_light[1],linewidth=4)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_yticks([])
xticks = ax.xaxis.get_major_ticks()
for i in tck:
    xticks[i].set_visible(False)


ax1=subplot(352,projection='polar')
plot(tcurv_dark[1],linewidth=4,color='grey')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_yticks([])
qxticks = ax1.xaxis.get_major_ticks()
for i in tck:
    xticks[i].set_visible(False)

ax2=subplot(356, projection='polar')
plot(tcurv_light[2],linewidth=4)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_yticks([])
xticks = ax2.xaxis.get_major_ticks()
for i in tck:
    xticks[i].set_visible(False)

ax3=subplot(357, projection='polar')
plot(tcurv_dark[2],linewidth=4,color='grey')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_yticks([])
xticks = ax3.xaxis.get_major_ticks()
for i in tck:
    xticks[i].set_visible(False)




























def f_ang_tcurves(cell_id,ep, spikes, position):
    '''computes the tcurve for a single cell each time the animal samples all angular bins
    inputs: cell_id= an integer corresponding to the cell column name in the tcurve data frame.
            ep= epoch (intervalSet)
            spikes= dictionary of spike times
            position=position['ry']
            
    outputs: fig
            peak_frate in each epoch
    '''
    ang_ep=full_ang(ep,position)
    eps=np.arange(len(ang_ep))
    fig=plt.figure()
    frates=pd.DataFrame(index=eps,columns=[0])
    sz=int(len(eps)/7)+1  
    for j,q in enumerate(eps): #epoch of interest 
        subplot(sz,7,j+1, projection='polar')
        f_ang_ep1=nts.IntervalSet(start=ang_ep.loc[j,'start'],end=ang_ep.loc[j,'end'])
        tcuve_f1=computeAngularTuningCurves(spikes,position,f_ang_ep1)
        frates.iloc[j,0]=tcuve_f1[cell_id].max()
        plt.plot(tcuve_f1[cell_id],label=str(j),color='k') 
        legend()
    return fig,frates
      
  

figure()
      
_,frates=f_ang_tcurves(1,wake_ep_2,spikes,position['ry'])
       
        
        
        for i,x in enumerate(cells):
            ax=subplot(6,7,j+1, projection='polar')
            plt.plot(tcuve_f1[x],label=str(j),color='k')
            xticks = ax.xaxis.get_major_ticks()
            xticks[1].label.set_visible(False)
            xticks[3].label.set_visible(False)
            xticks[5].label.set_visible(False)
            xticks[7].label.set_visible(False)
            ax.fill_between(tcuve_f1[x].index,tcuve_f1[x].values,0, color='darkgrey')
            ax.set_xticklabels([])
            legend()
        
    az=subplot(5,5,18,projection='polar')
    plt.plot(tuning_curves_2[7],color='k',zorder=2)
    fill_between(tuning_curves_2[7].index,tuning_curves_2[7].values,0, color='red',zorder=2)
fig.suptitle('#KA41(wildtype)_cell 19', fontsize=25)
#ax.grid(False)             
az.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])




ax=subplot(gs[p1,i],projection='polar')

gs = gridspec.GridSpec(3,3)
fig,ax=subplots()

for i,x in enumerate (cells):
    ax=subplot(gs[2,i], projection='polar')
    #ax=subplot(4,5,i+1,projection='polar')
    plt.plot(tuning_curves_1,label=str(i),color='darkgrey', linewidth=3)
    #legend()    
    a=[0,90,180,270]
    ax.set_thetagrids(a)#ang_direction
    ax.set_xticklabels(['E','N','W','S'])
    ax.set_yticks([])
    #ax.set_yticks([])  
    #ax.set_xticklabels([])
    #ax.xaxis.grid(linewidth=5)
    ax.tick_params(labelsize=23)

light_patch = mpatches.Patch(color='cyan', label='Light (10min)')
dark_patch=mpatches.Patch(color='darkgrey', label='Dark (10min)')
plt.legend(handles=[light_patch,dark_patch],loc='top right',bbox_to_anchor=(1.5,2.8),fontsize=30)    
plt.savefig('try', dpi=600, format='svg')

ax=subplot(projection='polar')
plot(tuning_curves_1[1])
a=[0,90,180,270]
ax.set_thetagrids(a)#ang_direction
ax.set_xticklabels(['E','N','W','S'])
ax.set_yticks([])
ax.set_rgrids([20])#vector

################################################################################
fig = plt.figure()
ax=subplot(339,projection='polar')
plt.plot(tuning_curves_1[0],label='Cell# 1',color='darkgrey', linewidth=3)
    #legend()    
a=[0,90,180,270]
ax.set_thetagrids(a)#ang_direction
ax.set_xticklabels(['E','N','W','S'])
ax.set_yticks([])
#ax.set_yticks([])  
#ax.set_xticklabels([])
#ax.xaxis.grid(linewidth=5)
ax.tick_params(labelsize=15)
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

ax1=subplot(331,projection='polar')
plt.plot(tuning_curves_2_ka30[9],label='Cell# 1',color='magenta', linewidth=3)
    #legend()    
a=[0,90,180,270]
ax1.set_thetagrids(a)#ang_direction
ax1.set_xticklabels(['E','N','W','S'])
ax1.set_yticks([])
ax.set_yticks([])  
ax.set_xticklabels([])
#ax.xaxis.grid(linewidth=5)
ax1.tick_params(labelsize=15)
###############################################################################


