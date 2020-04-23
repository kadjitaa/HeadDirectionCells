# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:12:06 2020

@author: kasum
"""

data_alo= pd.DataFrame(index=(['Egocentric', 'Alocentric']), columns=(['blind_t1','blind_t2','sight_t1', 'sight_t2'])) 
fig,ax=subplots()
ax1=bar(0,56,yerr=8.9,color='k')
ax2=bar(1, 15,yerr=11.5,color='k')
ax3=bar(2.5,46.17,yerr=7.3,color='none', edgecolor='k', linewidth=3)
ax4=bar(3.5,28.3,yerr=8.5,color='none', edgecolor='k', linewidth=3)
ax5=bar(5.5,38.5,yerr=10,color='k')
ax6=bar(6.5,36,yerr=11,color='k')
ax7=bar(8,62.8,yerr=6.6,color='white', edgecolor='k', linewidth=3)
ax8=bar(9,40.08,yerr=11.7,color='none', edgecolor='k', linewidth=3)
ax.set_ylim(0,80)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([])
ax.set_ylabel('Latency (s)',size=18)
ax.tick_params(labelsize=16)