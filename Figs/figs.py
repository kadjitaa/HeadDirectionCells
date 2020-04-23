# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:34:12 2020

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


data_files='C:/Users/kasum/Documents/HD_Drift/data'
dark_light=pd.read_hdf(data_files+'/dark_light_dataset.h5')



#1- Example HD cells in light and dark
l_tc=dark_light.loc[:,'light_tc'][0][0]
d_tc=dark_light.loc[:,'dark_tc'][0][0]
cells=[1,12,10]


ylabels=8.5




fig=figure(figsize=(11.48,5.59))
gs=GridSpec(3,4)
gs1=GridSpecFromSubplotSpec( 3,3,subplot_spec=gs[0]) 
for i,x in enumerate(cells):
    subplot(gs1[i,0],projection='polar')
    plot(l_tc[x],c='deepskyblue')
    remove_polarAx(gca(),True)
    gca().set_yticks([])
    subplot(gs1[i,1], projection='polar')
    plot(d_tc[x],c='k')
    remove_polarAx(gca(),True)
    gca().set_yticks([])

#2- Mean Firing Rates
####################################################
mfrate=dark_light.loc[:,'mfrates'][0]; mfrate=mfrate.reindex(columns=['dark','light'])
#####################################################

gs2=GridSpecFromSubplotSpec(3,6, subplot_spec=gs[1:3])
subplot(gs2[0:,0])
#boxplot(mfrate.T,showfliers=False)
plot(mfrate.T, c='grey', alpha=0.5)
plot(mfrate.mean(), c='r', linewidth=2)
gca().tick_params(labelsize=8.5,pad=0.4)
gca().set_xticklabels(['Light','Dark'],rotation=0,fontsize=10) #rotation_mode="anchor",ha='right')
#gca().set_ylim(-1,round((mfrate.values.max()),0)+2)
gca().set_ylabel('Mean firing rate (Hz)', fontsize=ylabels, labelpad=0.05)
gca().spines['bottom'].set_position(('axes',-0.05))
gca().spines['left'].set_position(('axes',-0.05))
remove_box()
plot([0,1],[34,34],c='k')
plt.annotate('n.s',[0.35,35], fontsize=7)


#3- Peak Firing Rates
##########################################################
pfrate=dark_light.loc[:,'pfrates'][0]; pfrate=pfrate.reindex(columns=['dark','light'])
##########################################################

subplot(gs2[0:,2])
#boxplot(mfrate.T,showfliers=False)
plot(pfrate.T, c='grey', alpha=0.5)
plot(pfrate.mean(), c='r', linewidth=2)
gca().tick_params(labelsize=8.5,pad=0.4)
gca().set_xticklabels(['Light','Dark'],rotation=0,fontsize=10) #rotation_mode="anchor",ha='right')
#gca().set_ylim(-1,round((mfrate.values.max()),0)+2)
gca().set_ylabel('Peak firing rate (Hz)', fontsize=ylabels, labelpad=0.05)
gca().spines['bottom'].set_position(('axes',-0.05))
gca().spines['left'].set_position(('axes',-0.05))
remove_box()
plot([0,1],[113,113],c='k')
plt.annotate('**',[0.35,114], fontsize=7)

#4- Stability
#########################################################
stab=dark_light.loc[:,'stability'][0]; stab=stab.reindex(columns=['dark','light'])
#########################################################

subplot(gs2[0:,4])
#boxplot(mfrate.T,showfliers=False)
plot(stab.T, c='grey', alpha=0.5)
plot(stab.mean(), c='r', linewidth=2)
gca().tick_params(labelsize=8.5,pad=0.4)
gca().set_xticklabels(['Light','Dark'],rotation=0,fontsize=10) #rotation_mode="anchor",ha='right')
gca().set_ylabel('Stability (r)', fontsize=ylabels, labelpad=0.05)
#gca().set_ylim(-1,round((mfrate.values.max()),0)+2)
gca().spines['bottom'].set_position(('axes',-0.05))
gca().spines['left'].set_position(('axes',-0.05))
gca().set_yticks([-1,0,1])
gca().set_ylim(-1.04, 1.07)
gca().set_yticklabels([-1,0,1])
remove_box()
plot([0,1],[1.05,1.05],c='k')
plt.annotate('**',[0.35,1.06], fontsize=7)

#5- Mean Vector Length
#########################################################
vlen=dark_light.loc[:,'vlength'][0]; vlen=vlen.reindex(columns=['dark','light'])
#########################################################
gs3=GridSpecFromSubplotSpec(3,6, subplot_spec=gs[4:6])
subplot(gs3[0:,0])
#boxplot(mfrate.T,showfliers=False)
plot(vlen.T, c='grey', alpha=0.5)
plot(vlen.mean(), c='r', linewidth=2)
gca().set_ylabel('Rayleigh vector', fontsize=ylabels, labelpad=0.05)
gca().tick_params(labelsize=8.5,pad=0.4)
gca().set_xticklabels(['Light','Dark'],rotation=0,fontsize=10) #rotation_mode="anchor",ha='right')
#gca().set_ylim(-1,round((mfrate.values.max()),0)+2)
gca().spines['bottom'].set_position(('axes',-0.05))
gca().spines['left'].set_position(('axes',-0.05))
gca().set_yticks([-1,0,1])
gca().set_ylim(-1.04, 1.07)
gca().set_yticklabels([-1,0,1])
remove_box()
plot([0,1],[1.05,1.05],c='k')
plt.annotate('**',[0.35,1.06], fontsize=7)


fig_dir='C:/Users/kasum/Dropbox/ADn_Project/paper1_figs'
plt.savefig(fig_dir+'/Fig1.svg',dpi=300, format='svg', bbox_inches="tight", pad_inches=0.05)
sys.exit()
#subplot(gs3[0:,2])



##############################################################################
##############################################################################
##Figs 2
##############################################################################
##############################################################################








#Stats
#p_val=round(scipy.stats.wilcoxon(stab['light'], stab['dark'])[1],5)






#plt.text(0.03,0.01,round(l_tc[x].values.max(),1))


sys.exit()
