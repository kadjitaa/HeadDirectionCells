# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 05:42:48 2020

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
light_cc=pd.read_hdf(data_files+'\light_cc.h5')



#1- Example HD cells in light and dark
#cells=[1,12,10]

fig=figure(figsize=(11.48,5.59))
gs=GridSpec(4,3)
gs2=GridSpecFromSubplotSpec(3,6, subplot_spec=gs[0:2])
subplot(gs2[0:,0])


######CrossCor Population Heatmap
#cell=pairs(tcurv_2)

merged_cc_light=light_cc

tmp=merged_cc_light[cell] #sorts the cross_corr mat based on the angular differences
tmp = tmp - tmp.mean(0)
tmp = tmp / tmp.std(0)
tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))
imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')

times = merged_cc_light.index.values
xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])])	
#yticks([0, len(cell)-1], [1, len(cell)], fontsize = 6)
gca().set_ylabel('Cell Pairs', fontsize=8)
gca().set_xticklabels([-5,0,5])
#gca().set_ylim(78,0)
#gca().set_yticks([0,20,40,60])
title('Light',fontweight='bold')
xlabel("Time lag (s)", fontsize = 8)
gca().tick_params(labelsize=8,pad=0.4)




subplot(gs2[0:,2])
dark_cc=pd.read_hdf(data_files+'\dark_cc.h5')

tmp=dark_cc[cell] #sorts the cross_corr mat based on the angular differences
tmp = tmp - tmp.mean(0)
tmp = tmp / tmp.std(0)
tmp = scipy.ndimage.gaussian_filter(tmp.T, (1, 1))

imshow(tmp, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
times = dark_cc.index.values
xticks([0, np.where(times==0)[0], len(times)], [int(times[0]), 0, int(times[-1])])	
title('Dark',fontweight='bold')
gca().set_xticklabels([-5,0,5])
#gca().set_ylim(78,0)
#gca().set_yticks([0,20,40,60])
gca().set_yticklabels([])
xlabel("Time lag (s)", fontsize = 8)
gca().tick_params(labelsize=8,pad=0.4)



fig_dir='C:/Users/kasum/Dropbox/ADn_Project/paper1_figs'
plt.savefig(fig_dir+'/Fig3.svg',dpi=300, format='svg', bbox_inches="tight", pad_inches=0.05)
