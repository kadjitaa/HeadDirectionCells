# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:20:17 2019

@author: kasum
"""
import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *
import math

import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


#Define Training Data & Epoch
tcurv= tcurv_train # tcurves for training decoder
Epoch= ep2 #epoch to be decoded

decoded_pos,ang=decodeHD(tcurv,spikes, Epoch) #run decoder

decoded_pos=pd.DataFrame(decoded_pos)
actual_pos=position['ry'].restrict(Epoch)


#######################################################
##Downsample Rotation Y
#######################################################

def makeBins(ep, bin_size=200): #the bin size is based on the bin size of the decoder
    bins_=  np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)      
    return bins_

bins = makeBins(Epoch)

index = np.digitize(actual_pos.as_units('ms').index.values, bins)-1

down_actual_pos = actual_pos.groupby(index).mean()  # here you are taking the mean of the positions corresponding to each unique binned index

down_actual_pos = pd.DataFrame(nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d =down_actual_pos.values[0:len(bins)-1], time_units = 'ms'))

#Compute Decoding Error
decoded_err=abs(np.arctan2(np.sin(down_actual_pos-decoded_pos),np.cos(down_actual_pos-decoded_pos)))
mean_decoded_err=print(np.abs(decoded_err).mean())


######################################################
##Downsample Position XZ
#####################################################
xypos=position[['x','z']].restrict(Epoch)


def makeBins(ep, bin_size=200): #the bin size is based on the bin size of the decoder
    bins_=  np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)      
    return bins_

bins = makeBins(Epoch)

index = np.digitize(xypos.as_units('ms').index.values, bins)-1

down_xypos = xypos.groupby(index).mean()  # here you are taking the mean of the positions corresponding to each unique binned index

down_xpos = pd.DataFrame(nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d =down_xypos.iloc[:,0].values[0:len(bins)-1], time_units = 'ms'))
down_ypos = pd.DataFrame(nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d =down_xypos.iloc[:,1].values[0:len(bins)-1], time_units = 'ms'))


##Merging Data
data=pd.DataFrame(index=down_actual_pos.index.values,columns=['ry','decoded_ry','x','y','err'])
data['ry']=down_actual_pos
data['decoded_ry']=decoded_pos
data['x']=down_xpos
data['y']=down_ypos
data['err']=decoded_err

conditions=['min','max','mean','std']
#PLOT DECODING ERROR projected unto x,y
#gs=GridSpec(1,4)
for i,x in enumerate(conditions):
    #subplot(gs[i])
    fig = plt.figure(figsize=(2.9622,1.8748,))
    stats,_,_,_=scipy.stats.binned_statistic_2d(data['x'],data['y'], data['err'], statistic=x,bins=30)
    stats[np.isnan(stats)]=0
    stats=gaussian_filter(stats,sigma=0.6)
    q=imshow(np.rot90(stats),cmap='jet',interpolation = 'bilinear')
    gca().set_yticks([])
    gca().set_xticks([])
    gca().set_ylabel('position Y')
    gca().set_xlabel('position X')
    #remove_box()
    cbar=fig.colorbar(q,orientation='vertical')
    #cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel('decoding error (rad)')

    fig.suptitle('GNAT_'+x+' Decoding Error')
    fig_dir='C:/Users/kasum/Dropbox/ADn_Project/decodingHeatmaps'
    plt.savefig(fig_dir+'/GNAT'+x+'.svg',dpi=300, format='svg', bbox_inches="tight", pad_inches=0.05)

    
    



fig = plt.figure()

#PLOTTING 3D
ax = fig.add_subplot(111, projection='3d')
X=data['x'].values
Y=data['y'].values
Z=data['err'].values

threshold=data['err'].values.max() *0.90
for i,x in enumerate(data['err']):
    v=x**4
    if x > threshold:
        ax.scatter(X[i],Y[i],Z[i], s=v,c='darkmagenta',alpha=0.5)

plot(X,Y,np.zeros(len(X)), c='grey')

gca().set_zlim(0,4)
gca().set_xlabel('Position X')
gca().set_ylabel('Position Y')
gca().set_zlabel('Decoding Error (rad)')

