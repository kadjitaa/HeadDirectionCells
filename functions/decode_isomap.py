# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:12:40 2020

@author: kasum
"""

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

##############################################
###Generate Ring Manifold
##############################################
ep=ep1
s=200 #bin_size

mfold_mat, mfold_t=makeRingManifold(spikes,ep1,position['ry'],bin_size=s)
mfold_mat=mfold_mat-[0.0113,0.03985]
mfold_t=pd.Series(mfold_t)

##############################################
###Downsampling actual position
##############################################
a_ang=position['ry'].restrict(ep)
bins = makeBins(ep, bin_size=s)
index = np.digitize(a_ang.as_units('ms').index.values, bins)-1

down_a_ang = a_ang.groupby(index).mean()  # here you are taking the mean of the positions corresponding to each unique binned index
down_a_ang = pd.DataFrame(nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d =down_a_ang.values[0:len(bins)-1], time_units = 'ms'))


#############################################
###Using isomap to decode position
#############################################
d_ang = np.arctan2(mfold_mat[:,1], mfold_mat[:,0])
decoded_ang = pd.Series((d_ang + 2*np.pi)%(2*np.pi))

decoded_ang=pd.DataFrame(nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d=decoded_ang.values[0:-1], time_units='ms'))



############################################
###Plotting
############################################
figure();plot(down_a_ang,color='k'); plot(decoded_ang,color='r')

flip_ang=(2*np.pi-decoded_ang); plot(flip_ang,color='g', linestyle='-')


flip_new_ang=flip_ang-((down_a_ang.values-flip_ang.values).mean())
plot(flip_new_ang)

#Compute Decoding Error
decoded_err=np.arctan2(np.sin(down_a_ang-decoded_ang),np.cos(down_a_ang-decoded_ang))
decoded_err=np.arctan2(np.sin(down_a_ang-flip_ang),np.cos(down_a_ang-flip_ang))
mean_decoded_err=print(np.abs(decoded_err).mean())

plt.figure(figsize=(16,7))

#plot(decoded_pos)
ax1=subplot(311)
plot(down_a_ang, linewidth=1.5)
ax1.set_ylim(0,2*np.pi)
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.set_ylabel('Head Direction (rad)', size=16)
ax1.tick_params(labelsize=15)

ax2=subplot(312)
plot(flip_ang, linewidth=1.5,color='r')

ax3=subplot(313)
plot(decoded_ang,color='r')

def computeAngularVelocity(spikes, angle, ep, nb_bins = 20, bin_size = 100000):
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))    
    tmp2             = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=30.0)        
    time_bins        = np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
    index             = np.digitize(tmp2.index.values, time_bins)
    tmp3             = tmp2.groupby(index).mean()
    tmp3.index         = time_bins[np.unique(index)-1]+50000
    tmp3             = nts.Tsd(tmp3)
    tmp4            = np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
    
    
    
embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(mfold_mat)

plt.scatter(embedding[:, 0], embedding[:, 1], s=0.1, cmap='Spectral');




