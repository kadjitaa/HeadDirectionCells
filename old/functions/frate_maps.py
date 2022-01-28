# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:21:26 2020

@author: kasum
"""
from functions import *
def frate_maps(spikes, position, ep,_bins=40):
    GF, ext = computePlaceFields(spikes, position[['x', 'z']], ep, _bins)
    fig,ax4 =subplots()
    #GF=GF.T
    #GF=GF.flip()
    sz=int(len(spikes.keys())/4)+1
    for i,k in enumerate(spikes.keys()):
       ax4=subplot(sz,4,i+1)
       tmp = gaussian_filter(GF[k].values,sigma = 1.5)
       #for i,v in enumerate(tmp):
           #for j,x in enumerate(tmp):    
               #if tmp[i][j] < 0:
                   #tmp[i][j]=NaN
       im=ax4.imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
      # plt.colorbar(im, cax = fig.add_axes([0.612, 0.535, 0.025, 0.17]))#   left/right  up/down  width height
       ax4.invert_yaxis()
       ax4.axis('off')
    return fig


