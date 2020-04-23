# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:12:17 2019

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

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


###############################################################################
##Example TCURVES + SPK_PATH PLOT
###############################################################################
gs = gridspec.GridSpec(3,4)

fig=figure()

ep=wake_ep_1; q=0
tcurve=tuning_curves_1; r=1

for i,x in enumerate(cells):
    ax=subplot(gs[i,q], projection='polar')
    plt.plot(tcurve[x],label='Cell# 1',color='k', linewidth=3)
    a=[0,90,180,270]
    ax.set_thetagrids(a)#ang_direction
    #ax.set_xticklabels(['E','N','W','S'])
    ax.set_yticks([])
    ax.set_yticks([])  
    ax.set_xticklabels([])
    #ax.xaxis.grid(linewidth=5)
    #ax.tick_params(labelsize=15)

for i,x in enumerate(cells):
    ax2=subplot(gs[i,r])
    path=ax2.plot(position['x'].restrict(ep),position['z'].restrict(ep),color='darkgrey', alpha=0.5)  
    pts=ax2.scatter(position['x'].realign(spikes[x].restrict(ep)),position['z'].realign(spikes[x].restrict(ep)),s=0.5,c='magenta',label='Cell# 1')         
    plt.axis('off')

cell1=plt.text(0.01, 0.765, 'Cell# 1', fontsize=15, transform=plt.gcf().transFigure)
cell2=plt.text(0.01, 0.5, 'Cell# 2', fontsize=15, transform=plt.gcf().transFigure)
cell3=plt.text(0.01, 0.235, 'Cell# 3', fontsize=15, transform=plt.gcf().transFigure)

light=plt.text(0.23, 0.92, 'Cylinder', fontsize=23, transform=plt.gcf().transFigure)
dark=plt.text(0.63, 0.92, 'Square', fontsize=23, transform=plt.gcf().transFigure)   




###################################################################################
#RATE MAP
##################################################################################
GF, ext = computePlaceFields(spikes, position[['x', 'z']], wake_ep_1, 70)
fig,ax4 =subplots()

#GF=GF.T
#GF=GF.flip()
for i,k in enumerate(spikes.keys()):
   ax4=subplot(4,2,i+1)
   tmp = gaussian_filter(GF[k].values,sigma = 2)
   #for i,v in enumerate(tmp):
       #for j,x in enumerate(tmp):    
           #if tmp[i][j] < 0:
               #tmp[i][j]=NaN
   im=ax4.imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
  # plt.colorbar(im, cax = fig.add_axes([0.612, 0.535, 0.025, 0.17]))#   left/right  up/down  width height
   ax4.invert_yaxis()
   ax4.axis('off')
show()


#######################################################################################
#OCCUPANCY HEATMAP
###############################################################################

#LIGHT
threshold=0.13; _bins=28  #50b
position= position
ep=wake_ep_1_ka30
occu=computeOccupancy(position.restrict(ep),_bins)
occu=gaussian_filter(occu,sigma=0.7)

fig, ax = plt.subplots()
#Light
ax=subplot(131)
'''for i,z in enumerate(occu):
    for x,y in enumerate(occu):
        if occu[i][x] <=threshold:
            occu[i][x]=NaN'''
q=imshow(occu,cmap='jet',interpolation = 'bilinear')
ax.axis('off')
ax.set_title('LIGHT', size=15)
#gca().invert_yaxis()
#cbar=fig.colorbar(q,orientation='horizontal',cax=fig.add_axes([0.80, 0.07, 0.15, 0.06])) #left/right  up/down  width height
#cbar.set_label('min        max',size=15)
#cbar.set_ticks([])
    #cbar.ax.set_xlabel('occu')

#DARK
threshold=0.13; _bins=28  #50b
position= position_ka30
ep=wake_ep_2_ka30
occu=computeOccupancy(position.restrict(ep),_bins)
occu=gaussian_filter(occu,sigma=0.7)

#Light
ax1=subplot(132)
'''for i,z in enumerate(occu):
    for x,y in enumerate(occu):
        if occu[i][x] <=threshold:
            occu[i][x]=NaN'''
q=imshow(occu,cmap='jet',interpolation = 'bilinear')
ax1.axis('off')
ax1.set_title('DARK', size=15)

#CG
threshold=0.13; _bins=28  #50b
position= position_ka43
ep=wake_ep_3
occu=computeOccupancy(position.restrict(ep),_bins)
occu=gaussian_filter(occu,sigma=0.7)

#Light
ax2=subplot(133)
'''for i,z in enumerate(occu):
    for x,y in enumerate(occu):
        if occu[i][x] <=threshold:
            occu[i][x]=NaN'''
q=imshow(occu,cmap='jet',interpolation = 'bilinear')
ax2.axis('off')
ax2.set_title('Cg BLIND', size=15)
#gca().invert_yaxis()
cbar=fig.colorbar(q,orientation='horizontal',cax=fig.add_axes([0.80, 0.075, 0.15, 0.06])) #left/right  up/down  width height
cbar.set_label('min        max',size=13)
cbar.set_ticks([])

#Dark
ax1=subplot(1,1,1)
ep=wake_ep_2_ka30
occu=computeOccupancy(position.restrict(ep),_bins)

'''for i,z in enumerate(occu):
    for x,y in enumerate(occu):
        if occu[i][x] <=threshold:
            occu[i][x]=NaN'''
r=imshow(occu,cmap='hot',interpolation = 'bilinear')
gca().invert_yaxis()

ax1.axis('off')
cbar=fig.colorbar(r,orientation='horizontal',cax=fig.add_axes([0.84, 0.05, 0.15, 0.06])) #left/right  up/down  width height
cbar.set_label('min            max',size=15)
cbar.set_ticks([])

#min=plt.text(0.44,0.27,'min',size=20,transform=plt.gcf().transFigure)
#max=plt.text(0.56,0.27,'max',size=20,transform=plt.gcf().transFigure)

light=plt.text(0.21, 0.92, 'Cg Blind Occupancy', fontsize=23, transform=plt.gcf().transFigure)
dark=plt.text(0.72, 0.92, 'DARK', fontsize=23, transform=plt.gcf().transFigure)   



##############################################################################
#DECODING
##############################################################################
tcurv= tuning_curves_1_train # tcurves for training decoder
Epoch=wake_ep_1_ka30#epoch to be decoded
position=position_ka30

decoded_pos,ang=decodeHD(tcurv,spikes_ka30, Epoch) #run decoder
decoded_pos=pd.DataFrame(decoded_pos)
actual_pos=position['ry'].restrict(Epoch)


def makeBins(ep, bin_size=200):        
    return np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)

bins = makeBins(Epoch)
index = np.digitize(actual_pos.as_units('ms').index.values, bins)-1
down_actual_pos = actual_pos.groupby(index).mean()
down_actual_pos = pd.DataFrame(nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d =down_actual_pos.values[0:len(bins)-1], time_units = 'ms'))

#Compute Decoding Error
decoded_err=np.arctan2(np.sin(down_actual_pos-decoded_pos),np.cos(down_actual_pos-decoded_pos))

mean_decoded_err=np.abs(decoded_err).mean()

#decoded_err2=np.arctan2(np.sin(abs(down_actual_pos-decoded_pos)), np.cos(abs(down_actual_pos-decoded_pos)))
#mean_decoded_err2=decoded_err2.mean()

wake_ep_1_l2=nts.IntervalSet(start=wake_ep_ka43.loc[0,'end'] - 1.2e+8,end = wake_ep_ka43.loc[0,'end'])
#wake_ep_2_l2=nts.IntervalSet(start=wake_ep_ka30.loc[3,'end'] - 1.2e+8,end = wake_ep_ka30.loc[3,'end'])

#first 2mins
ac_pos_f2_idx=down_actual_pos.index < (down_actual_pos.index[0]+1.2e+8)
ac_pos_f2=down_actual_pos[ac_pos_f2_idx]
dec_pos_f2=decoded_pos[ac_pos_f2_idx]


##final 2mins
ac_pos_l2_idx=down_actual_pos.index > (down_actual_pos.index[-1]- 1.2e+8) 
ac_pos_l2=down_actual_pos[ac_pos_l2_idx]
dec_pos_l2=decoded_pos[ac_pos_l2_idx]

plt.figure(figsize=(16,7))

#plot(decoded_pos)
ax1=subplot(321)
plot(ac_pos_f2, linewidth=1.5);plot(dec_pos_f2, linewidth=1.5,color='r')
ax1.set_ylim(0,2*np.pi)
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.set_ylabel('Head Direction (rad)', size=14)
ax1.tick_params(labelsize=15)

ax2=subplot(322)
plot(ac_pos_l2, linewidth=1.5);plot(dec_pos_l2,linewidth=1.5,color='r')
ax2.set_ylim(0,2*np.pi)
ax2.set_xticklabels([])
ax2.set_xticks([])
ax2.set_yticklabels([])
ax2.tick_params(labelsize=15)
##################################################################################

tcurv= tuning_curves_2_train # tcurves for training decoder
Epoch=wake_ep_2_ka30#epoch to be decoded
position=position_ka30


decoded_pos,ang=decodeHD(tcurv,spikes_ka30, Epoch) #run decoder
decoded_pos=pd.DataFrame(decoded_pos)
actual_pos=position['ry'].restrict(Epoch)


def makeBins(ep, bin_size=200):        
    return np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)

bins = makeBins(Epoch)
index = np.digitize(actual_pos.as_units('ms').index.values, bins)-1
down_actual_pos = actual_pos.groupby(index).mean()
down_actual_pos = pd.DataFrame(nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d =down_actual_pos.values[0:len(bins)-1], time_units = 'ms'))

#Compute Decoding Error
decoded_err=np.arctan2(np.sin(down_actual_pos-decoded_pos),np.cos(down_actual_pos-decoded_pos))

mean_decoded_err=np.abs(decoded_err).mean()

#decoded_err2=np.arctan2(np.sin(abs(down_actual_pos-decoded_pos)), np.cos(abs(down_actual_pos-decoded_pos)))
#mean_decoded_err2=decoded_err2.mean()

wake_ep_1_l2=nts.IntervalSet(start=wake_ep_ka43.loc[0,'end'] - 1.2e+8,end = wake_ep_ka43.loc[0,'end'])
#wake_ep_2_l2=nts.IntervalSet(start=wake_ep_ka30.loc[3,'end'] - 1.2e+8,end = wake_ep_ka30.loc[3,'end'])

#first 2mins
ac_pos_f2_idx=down_actual_pos.index < (down_actual_pos.index[0]+1.2e+8)
ac_pos_f2=down_actual_pos[ac_pos_f2_idx]
dec_pos_f2=decoded_pos[ac_pos_f2_idx]


##final 2mins
ac_pos_l2_idx=down_actual_pos.index > (down_actual_pos.index[-1]- 1.2e+8) 
ac_pos_l2=down_actual_pos[ac_pos_l2_idx]
dec_pos_l2=decoded_pos[ac_pos_l2_idx]

#plot(decoded_pos)
ax1=subplot(323)
plot(ac_pos_f2, linewidth=1.5);plot(dec_pos_f2, linewidth=1.5,color='r')
ax1.set_ylim(0,2*np.pi)
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.set_ylabel('Head Direction (rad)', size=14)
ax1.tick_params(labelsize=15)

ax2=subplot(324)
plot(ac_pos_l2, linewidth=1.5);plot(dec_pos_l2,linewidth=1.5,color='r')
ax2.set_ylim(0,2*np.pi)
ax2.set_xticklabels([])
ax2.set_xticks([])
ax2.set_yticklabels([])
ax2.tick_params(labelsize=15)



tcurv= tuning_curves_3_train # tcurves for training decoder
Epoch=wake_ep_3#epoch to be decoded
position=position_ka43

decoded_pos,ang=decodeHD(tcurv,spikes_ka43, Epoch) #run decoder
decoded_pos=pd.DataFrame(decoded_pos)
actual_pos=position['ry'].restrict(Epoch)


def makeBins(ep, bin_size=200):        
    return np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)

bins = makeBins(Epoch)
index = np.digitize(actual_pos.as_units('ms').index.values, bins)-1
down_actual_pos = actual_pos.groupby(index).mean()
down_actual_pos = pd.DataFrame(nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d =down_actual_pos.values[0:len(bins)-1], time_units = 'ms'))


#Params
#first 2mins
ac_pos_f2_idx=down_actual_pos.index < (down_actual_pos.index[0]+1.2e+8)
ac_pos_f2=down_actual_pos[ac_pos_f2_idx]
dec_pos_f2=decoded_pos[ac_pos_f2_idx]


##final 2mins
ac_pos_l2_idx=down_actual_pos.index > (down_actual_pos.index[-1]- 1.2e+8) 
ac_pos_l2=down_actual_pos[ac_pos_l2_idx]
dec_pos_l2=decoded_pos[ac_pos_l2_idx]


##Dark_ep
ax3=subplot(325)
plot(ac_pos_f2, linewidth=1.5);plot(dec_pos_f2, linewidth=1.5,color='r')
ax3.set_ylim(0,2*np.pi)
ax3.set_xticklabels([])
ax3.set_xticks([])
ax3.set_ylabel('Head Direction (rad)', size=14)
ax3.tick_params(labelsize=15)

ax4=subplot(326)
plot(ac_pos_l2, linewidth=1.5);plot(dec_pos_l2,linewidth=1.5,color='r')
ax4.set_ylim(0,2*np.pi)
ax4.set_xticklabels([])
ax4.set_xticks([])
ax4.set_yticklabels([])
ax4.tick_params(labelsize=15)


act_patch = mpatches.Patch(color='#1f77b4', label='Actual HD')
dec_patch=mpatches.Patch(color='r', label='Decoded HD')
plt.legend(handles=[act_patch,dec_patch],loc='top right',bbox_to_anchor=(1.3,2.1),fontsize=13)

plt.subplots_adjust(top=0.885,bottom=0.035,left=0.125,right=0.885,hspace=0.2,wspace=0.05)

light=plt.text(0.009, 0.76, 'LIGHT', fontsize=23, transform=plt.gcf().transFigure)
dark=plt.text(0.009, 0.3, 'DARK', fontsize=23, transform=plt.gcf().transFigure)   
firs2min=plt.text(0.13, 0.895, 'SESSION: first 2mins', fontsize=16, transform=plt.gcf().transFigure) 
last2min=plt.text(0.52, 0.895, 'SESSION: last 2mins', fontsize=16, transform=plt.gcf().transFigure) 




from scipy.ndimage import gaussian_filter

def occupancy_heat()
threshold=0.13; _bins=28  #50b
position= position
ep=wake_ep_1_ka30
occu=computeOccupancy(position.restrict(ep),_bins)
occu=gaussian_filter(occu,sigma=0.7)

fig, ax = plt.subplots()
#Light
ax=subplot(131)
q=imshow(occu,cmap='jet',interpolation = 'bilinear')
ax.axis('off')