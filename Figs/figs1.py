# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 01:40:25 2020

@author: kasum
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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
import statsmodels.api as sm

ang_ep=full_ang(ep1,positions['ry'])


#1- Example HD cells in light and dark
#cells=[1,12,10]

fig=figure(figsize=(11.48,5.59))
gs=GridSpec(4,4)

#cell1
gs4=GridSpecFromSubplotSpec( 6,6,subplot_spec=gs[0:2,0:])

subplot(gs4[0:2,0], projection ='polar')
plot(tcurv_2[1],linewidth=2.5,c='deepskyblue')
remove_polarAx(gca(),True)
gca().set_yticks([])
title('LIGHT')

subplot(gs4[0:2,1], projection ='polar')
plot(tcurv_1[1],linewidth=2.5,c='k')
remove_polarAx(gca(),True)
gca().set_yticks([])
title('DARK')


#cell2
ax2=subplot(gs4[2:4,0], projection='polar')
plot(tcurv_2[2],linewidth=2.5,c='deepskyblue')
remove_polarAx(gca(),True)
gca().set_yticks([])

ax3=subplot(gs4[2:4,1], projection='polar')
plot(tcurv_1[2],linewidth=2.5,c='k')
remove_polarAx(gca(),True)
gca().set_yticks([])




###Drift in cells
gs2=GridSpecFromSubplotSpec( 6,6,subplot_spec=gs[0:2,1:])
   
eps=np.arange(len(ang_ep))
pairs=[1,2,3,4,21]#21
cells_i=[1]
for s,q in enumerate(cells_i):
    cells=[q]
    #fig.suptitle('#Cell_' +str(q),fontsize=25)
    for j,q in enumerate(pairs): #epoch of interest 
        f_ang_ep1=nts.IntervalSet(start=ang_ep.loc[j,'start'],end=ang_ep.loc[j,'end'])
        tcuve_f1=computeAngularTuningCurves(spikess,positions['ry'],f_ang_ep1)   
        for i,x in enumerate(cells):
            subplot(gs2[0:2,j+1], projection='polar')
            title('Epoch'+str(j),fontsize=10)
            plt.plot(tcuve_f1[x],color='k',alpha=0.75)
            remove_polarAx(gca(),True)
            gca().set_yticks([])


 
cells_i=[2]
for s,q in enumerate(cells_i):
    cells=[q]
    #fig.suptitle('#Cell_' +str(q),fontsize=25)
    for j,q in enumerate(pairs): #epoch of interest 
        f_ang_ep1=nts.IntervalSet(start=ang_ep.loc[j,'start'],end=ang_ep.loc[j,'end'])
        tcuve_f1=computeAngularTuningCurves(spikess,positions['ry'],f_ang_ep1)   
        for i,x in enumerate(cells):
            subplot(gs2[2:4,j+1], projection='polar')
            plt.plot(tcuve_f1[x],color='k',alpha=0.75)
            remove_polarAx(gca(),True)
            gca().set_yticks([])
            

#permits annotation outside polar ax
#plt.text(0.02, 0.5, 'hivvvvvvc', fontsize=14, transform=plt.gcf().transFigure)


###CROSS CORROLOGRAMS--SIMILAR TUNING##############
            
'''lIGHT'''
data_files='C:/Users/kasum/Documents/HD_Drift/data'
data=pd.read_hdf(data_files+'\dark_light_dataset.h5')

gs4=GridSpecFromSubplotSpec( 3,3,subplot_spec=gs[2:4,0]) 
subplot(gs4[1:,0:])

cc=data.loc[:,'light_cc'][0][(5,8)]
plot(cc,color='dimgrey',linewidth=2)
plt.fill_between(cc.index,cc.values,0, color='dimgrey')
remove_box()
gca().set_ylim(0,round(cc.values.max(),0))
gca().set_xlim(-5000,5000)
gca().set_ylim(0,4)
gca().set_xticks([-5000,0,5000]); gca().set_xticklabels([-5,0,5])
gca().set_ylabel('Cross correlation',size=12,labelpad=0.5)
gca().set_xlabel('Time lag(s)',size=12)
gca().tick_params(labelsize=12,pad=0.4)

#example tcurves
#gs3=GridSpecFromSubplotSpec(3,2, subplot_spec=gs[1,0])
ax=subplot(gs4[0], projection='polar')
plot(tcurv_2[5])
plot(tcurv_2.index,tcurv_2[8].values*8)
remove_polarAx(gca(),True)
gca().set_yticks([])




'''Dark'''
gs5=GridSpecFromSubplotSpec( 3,3,subplot_spec=gs[2:4,1]) 
subplot(gs5[1:,0:])

data_files='C:/Users/kasum/Documents/HD_Drift/data'
data=pd.read_hdf(data_files+'\dark_light_dataset.h5')

cc=data.loc[:,'dark_cc'][0][(5,8)]
plot(cc,color='dimgrey',linewidth=2)
plt.fill_between(cc.index,cc.values,0, color='dimgrey')
remove_box()
gca().set_ylim(0,round(cc.values.max(),0))
gca().set_xlim(-5000,5000)
gca().set_ylim(0,4)
gca().set_xticks([-5000,0,5000]); gca().set_xticklabels([-5,0,5])
#gca().set_ylabel('Norm. correlation',size=12)
gca().set_xlabel('Time lag(s)',size=12)
gca().tick_params(labelsize=12,pad=0.4)

ax=subplot(gs5[0], projection='polar')
plot(tcurv_1[5])
plot(tcurv_1.index,tcurv_1[8].values*29)
remove_polarAx(gca(),True)
gca().set_yticks([])




###CROSS CORROLOGRAMS--OPPOSITE TUNING##############

cc=qwik_cc(1,2, spikes, wake_ep_2)

            
'''lIGHT'''
#data_files='C:/Users/kasum/Documents/HD_Drift/data'
#data=pd.read_hdf(data_files+'\dark_light_dataset.h5')

gs6=GridSpecFromSubplotSpec( 3,3,subplot_spec=gs[2:4,2]) 
subplot(gs6[1:,0:])

#cc=data.loc[:,'light_cc'][0][(8,10)]
plot(cc,color='dimgrey',linewidth=2)
plt.fill_between(cc.index,cc.values,0, color='dimgrey')
remove_box()
gca().set_ylim(0,0.8)
gca().set_xlim(-1000,1000)
gca().set_xticks([-1000,0,1000]); gca().set_xticklabels([-5,0,5])
#gca().set_ylabel('Cross correlation',size=10,labelpad=0.05)
gca().set_xlabel('Time lag(s)',size=12)
gca().tick_params(labelsize=12,pad=0.4)

#example tcurves
#gs3=GridSpecFromSubplotSpec(3,2, subplot_spec=gs[1,0])
ax=subplot(gs6[0], projection='polar')
plot(tuning_curves_2[1].index,tuning_curves_2[1].values*1.5,c='darkred')
plot(tuning_curves_2.index,tuning_curves_2[2].values,c='indianred')
remove_polarAx(gca(),True)
gca().set_yticks([])


'''Dark'''
cc=qwik_cc(1,2, spikes, wake_ep_1)

gs5=GridSpecFromSubplotSpec( 3,3,subplot_spec=gs[2:4,3]) 
subplot(gs5[1:,0:])

#data_files='C:/Users/kasum/Documents/HD_Drift/data'
#data=pd.read_hdf(data_files+'\dark_light_dataset.h5')

#cc=data.loc[:,'dark_cc'][0][(6,10)]
plot(cc,color='dimgrey',linewidth=2)
plt.fill_between(cc.index,cc.values,0, color='dimgrey')
remove_box()
gca().set_ylim(0,0.8)
gca().set_xlim(-1000,1000)
gca().set_xticks([-1000,0,1000]); gca().set_xticklabels([-5,0,5])
#gca().set_ylabel('Norm. correlation',size=12)
gca().set_xlabel('Time lag(s)',size=12)
gca().tick_params(labelsize=12,pad=0.4)

ax=subplot(gs5[0], projection='polar')
plot(tuning_curves_1[1].index,tuning_curves_1[1].values*3.5,c='darkred')
plot(tuning_curves_1.index,tuning_curves_1[2].values, c='indianred')
remove_polarAx(gca(),True)
gca().set_yticks([])


fig_dir='C:/Users/kasum/Dropbox/ADn_Project/paper1_figs'
plt.savefig(fig_dir+'/Fig2.svg',dpi=300, format='svg', bbox_inches="tight", pad_inches=0.05)


sys.exit()







###CROSS CORROLOGRAMS





top=0.893,
bottom=0.096,
left=0.038,
right=0.983,
hspace=0.15,
wspace=0.27