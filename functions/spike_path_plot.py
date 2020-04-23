# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:31:32 2020

@author: kasum
"""

def path_spk_plot(ep,spikes,position):
    """generates spikes superimposed on path plots for each cell """
    sz=int(len(spikes.keys())/4)+1
    fig = figure(figsize = (15,16))
    fig.suptitle('Spikes + Path Plot',size=30)
    for i in spikes:
        ax=subplot(sz,4,i+1)
        scatter(position['x'].realign(spikes[i].restrict(ep)),position['z'].realign(spikes[i].restrict(ep)),s=5,c='magenta',label=str(i))
        legend()
        plot(position['x'].restrict(ep),position['z'].restrict(ep),color='darkgrey', alpha=0.5)  
    return fig,ax

