# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:53:51 2020

@author: kasum
"""

def occu_heatmp(ep,position,_bins=50, threshold=0.13):
    """Generates an occupancy heat map"""
    
    occu=computeOccupancy(position.restrict(ep),_bins)
    occu=gaussian_filter(occu,sigma=0.7)
    for i,z in enumerate(occu):
        for x,y in enumerate(occu):
            if occu[i][x] <=threshold:
                occu[i][x]=NaN
    fig, ax = plt.subplots()
    q=ax.imshow(occu,cmap='jet',interpolation = 'bilinear')
    cbar=fig.colorbar(q,orientation='vertical')
    cticks=cbar.ax.get_xticks()
    cbar.set_ticks([])
    #cbar.set_ticklabels(['min','max'])
    #cbar.ax.set_xlabel('occu')
    ax.axis('off')