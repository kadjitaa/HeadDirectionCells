# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:51:08 2019

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
########################################
#Define params
########################################
ep=wake_ep_1
spks=spikes
pos= position

#######################################################################
#Rate maps
#######################################################################
GF, ext = computePlaceFields(spks, pos[['x', 'z']], ep, 70)
fig,ax4 =subplots()

#GF=GF.T
#GF=GF.flip()
for i,k in enumerate(spks.keys()):
   ax4=subplot(3,3,i+1)
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

###########################################################################
#Raw Rate Map (Spike+Path)
###########################################################################
#fig = figure(figsize = (15,16))
#fig.suptitle('Spikes + Path Plot',size=30)
fig,ax =subplots()

for i in spks:
    ax=subplot(3,3,i+1)
    scatter(pos['x'].realign(spks[i].restrict(ep)),pos['z'].realign(spks[i].restrict(ep)),s=5,c='magenta',label=str(i))
    legend()
    plot(pos['x'].restrict(ep),pos['z'].restrict(ep),color='darkgrey', alpha=0.5)  
