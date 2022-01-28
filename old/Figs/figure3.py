# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:55:22 2020

@author: kasum
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 04:32:46 2020

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


ang_ep=full_ang(ep1,position['ry'])


fig=figure(figsize=(11.48,5.59))

gs=GridSpec(2,4)
#plt.tight_layout(pad=1, w_pad=0.1, h_pad=1) #'''fixes fig layout'''

#tcurve light
    
gs1=GridSpecFromSubplotSpec( 2,2,subplot_spec=gs[0])
ax=subplot(gs1[0], projection='polar')
plot(tcurv_2[1])
remove_polarAx(gca(),True)
gca().set_yticks([])
title('LIGHT')


ax1=subplot(gs1[1], projection='polar')
plot(tcurv_1[1])
remove_polarAx(gca(),True)
gca().set_yticks([])
title('DARK')


ax2=subplot(gs1[2], projection='polar')
plot(tcurv_2[2])
remove_polarAx(gca(),True)
gca().set_yticks([])


ax3=subplot(gs1[3], projection='polar')
plot(tcurv_1[2])
remove_polarAx(gca(),True)
gca().set_yticks([])


 
gs2=GridSpecFromSubplotSpec( 2,11,subplot_spec=gs[0,:])
   
ang_ep=full_ang(wake_ep_2,position['ry'])
eps=np.arange(len(ang_ep))
pairs=[0,1,2,3,4,5,6,7,8,9,10]
cells_i=[3]

#fig.suptitle('#Cell_' +str(q),fontsize=25)
for j,k in enumerate(pairs): #epoch of interest 
    f_ang_ep1=nts.IntervalSet(start=ang_ep.loc[k,'start'],end=ang_ep.loc[k,'end'])
    tcuve_f1=computeAngularTuningCurves(spikes,position['ry'],f_ang_ep1)   
    for i,x in enumerate(cells):
        subplot(gs2[0,j], projection='polar')
        title('Epoch'+str(j),fontsize=9)
        plt.plot(tcuve_f1[x],color='grey',linewidth=3)
        remove_polarAx(gca(),True)
        gca().set_yticks([])

ang_ep=full_ang(wake_ep_1,position['ry'])
eps=np.arange(len(ang_ep))

#fig.suptitle('#Cell_' +str(q),fontsize=25)
for j,q in enumerate(pairs): #epoch of interest 
    f_ang_ep1=nts.IntervalSet(start=ang_ep.loc[q,'start'],end=ang_ep.loc[q,'end'])
    tcuve_f1=computeAngularTuningCurves(spikes,position['ry'],f_ang_ep1)   
    for i,x in enumerate(cells):
        subplot(gs2[1,j], projection='polar')
        plt.plot(tcuve_f1[x],color='grey',linewidth=3)
        remove_polarAx(gca(),True)
        gca().set_yticks([])


gs3=GridSpecFromSubplotSpec(3,2, subplot_spec=gs[1,0])
ax=subplot(gs3[0], projection='polar')
plot(tcurv_2[5])
plot(tcurv_2.index,tcurv_2[8].values*8)
remove_polarAx(gca(),True)
gca().set_yticks([])


###CROSS CORROLOGRAMS
data_files='C:/Users/kasum/Documents/HD_Drift/data'
data=pd.read_hdf(data_files+'\dark_light_dataset.h5')
c='k'
cc=data.loc[:,'light_cc'][0][(5,8)]

#gs4=GridSpecFromSubplotSpec(2,2, subplot_spec=gs[2,0])
subplot(gs3[1:,0:])
plot(cc,color=c,linewidth=3)
#plt.fill_between(cc.index,cc.values,0, color=c)
remove_box()
gca().set_ylim(0,round(cc.values.max(),0))
gca().set_xlim(-5000,5000)
gca().set_xticks([-5000,0,5000]); gca().set_xticklabels([-5,0,5])
gca().set_ylabel('Cross correlation',size=12)
gca().set_xlabel('Time lag(s)',size=12)
gca().tick_params(labelsize=12)
title('Light')


gs4=GridSpecFromSubplotSpec(3,2, subplot_spec=gs[5:6])
ax=subplot(gs4[0], projection='polar')
plot(tcurv_1[5])
plot(tcurv_1.index,tcurv_1[8].values*29)
remove_polarAx(gca(),True)
gca().set_yticks([])


###CROSS CORROLOGRAMS
#data_files='C:/Users/kasum/Documents/HD_Drift/data'
#data=pd.read_hdf(data_files+'\dark_light_dataset.h5')
c='darkgrey'
cc=dark_cc[(5,8)]

gs5=GridSpecFromSubplotSpec(3,2, subplot_spec=gs[5:6])
subplot(gs5[1:,0:2])
plot(cc,color=c)
plt.fill_between(cc.index,cc.values,0, color=c)
remove_box()
gca().set_ylim(0,round(cc.values.max(),0))
gca().set_xlim(-5000,5000)
gca().set_xticks([-5000,0,5000]); gca().set_xticklabels([-5,0,5])
#gca().set_ylabel('Norm. correlation',size=12)
gca().set_xlabel('Time lag(s)',size=12)
gca().tick_params(labelsize=12)
title('Dark')

######CrossCor Population Heatmap
#cell=pairs(tcurv_2)

merged_cc_light=light_cc

gs6=GridSpecFromSubplotSpec(1,3, subplot_spec=gs[5:7])

subplot(gs6[2])
tmp=merged_cc_light[cell] #sorts the cross_corr mat based on the angular differences

tmp = tmp - tmp.mean(0)
tmp = tmp / tmp.std(0)
tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))

imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')

times = merged_cc_light.index.values
xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])], fontsize = 14)	
#yticks([0, len(cell)-1], [1, len(cell)], fontsize = 6)
gca().set_ylabel('Cell Pairs', fontsize=12)
gca().set_xticklabels([-5,0,5])
gca().set_ylim(78,0)
gca().set_yticks([0,20,40,60,78])
title('Light',fontweight='bold')
xlabel("Time lag (s)", fontsize = 12)
gca().tick_params(labelsize=12)


gs7=GridSpecFromSubplotSpec(1,3 ,subplot_spec=gs[7])
ay=subplot(gs7[0,0:2])
tmp=dark_cc[cell] #sorts the cross_corr mat based on the angular differences
tmp = tmp - tmp.mean(0)
tmp = tmp / tmp.std(0)
tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))


imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')

times = dark_cc.index.values
xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])], fontsize = 14)	
yticks([])
title('Dark',fontweight='bold')
gca().set_xticklabels([-5,0,5])
gca().set_yticklabels([])
gca().set_ylim(78,0)
xlabel("Time lag (s)", fontsize = 12)
gca().tick_params(labelsize=12)

sys.exit()
d_loc=plt.text(8,0.4, 'DARK', size=16)
l_loc=plt.text(-.9,0.185, 'LIGHT', rotation='vertical',size=16)

a_loc=plt.text(-.1,0.43, '(a)',size=12,fontweight='bold')
b_loc=plt.text(1.30,0.23, '(b)',size=12,fontweight='bold')
c_loc=plt.text(8.64,0.23, '(c)',size=12,fontweight='bold')
d_loc=plt.text(-.9,0.0730, '(d)',size=12,fontweight='bold')




fig_dir='C:/Users/kasum/Dropbox/ADn_Project/paper1_figs'
plt.savefig(fig_dir+'/Figure2.svg',dpi=600, format='svg', bbox_inches="tight", pad_inches=0.05)





