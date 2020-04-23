# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:14:24 2019

@author: kasum
"""

gs = gridspec.GridSpec(8, 5)
#fig=plt.figure()


#ix=[0,1,2,3,4]
ix=3
p1=6
p2=7
#Light
for i in range(len(light_ep)):
    new_ep=nts.IntervalSet(start=light_ep.loc[i,'start'], end =light_ep.loc[i,'end'])
    tcurv=computeAngularTuningCurves(spikes,position ['ry'],new_ep,60)
    cell=tcurv[ix]  
    ax=subplot(gs[p1,i],projection='polar')
    #ax.set_facecolor('yellow')  
    ax.plot(cell, color='black')
    ax.fill_between(cell.index,cell.values,0, color='greenyellow')


#Dark    
for i in range(len(dark_ep)):
    new_ep=nts.IntervalSet(start=dark_ep.loc[i,'start'], end =dark_ep.loc[i,'end'])
    tcurv=computeAngularTuningCurves(spikes,position ['ry'],new_ep,60)
    cell=tcurv[ix]  
    ax1=subplot(gs[p2,i],projection='polar')
    #ax.set_facecolor('lightgrey')  
    ax1.plot(cell, color='black', alpha=10)
    #fig.subplots_adjust(left=0.1, bottom=0.2, right=0.8, top=1.3)  #the higher u want it to go the bigger the val for top-several other parameters that can help you adjust the arrangement
    ax1.fill_between(cell.index,cell.values,0, color='grey',alpha=1)
                                                           #the lower the left val the more closer it gets to the edge
#fig.suptitle('KA30_ADn HD Sliding Win Tcurves [3/3]', fontsize=25)
#light_patch = mpatches.Patch(color='greenyellow', label='Light (2m sWin)')
#dark_patch=mpatches.Patch(color='grey', label='Dark (2m sWin)')
#plt.legend(handles=[light_patch,dark_patch],loc='top right',bbox_to_anchor=(2.6,3.2),fontsize=18)

###############################################################################################################################################
                                                           
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


gs = gridspec.GridSpec(4,2)
fig,ax=subplots()

for i,x in enumerate (cells_1):
    ax=subplot(gs[i,0], projection='polar')
    plt.plot(all_tcurves[all_tcurves['animal_id']=='KA43']['tcurves'][0][x],label=str(i),color='k', linewidth=3)  
    a=[0,90,180,270]
    ax.set_thetagrids(a)#ang_direction
    ax.set_xticklabels(['E','N','W','S'])
    ax.set_yticks([])
    ax.tick_params(labelsize=15)

for i,x in enumerate (cells_1):
    ax=subplot(gs[i,1], projection='polar')
    plt.plot(all_tcurves[all_tcurves['animal_id']=='KA46']['tcurves'][1][x],label=str(i),color='magenta', linewidth=3)
    a=[0,90,180,270]
    ax.set_thetagrids(a)#ang_direction
    ax.set_xticklabels(['E','N','W','S'])
    ax.set_yticks([])
    ax.tick_params(labelsize=15)
    
    
for i,x in enumerate (cells_3):
    az=subplot(gs[3,i], projection='polar')
    plt.plot(all_tcurves[all_tcurves['animal_id']=='KA43']['tcurves'][0][x],label=str(i),color='k', linewidth=3)
    a=[0,90,180,270]
    az.set_thetagrids(a)#ang_direction
    az.set_xticklabels(['E','N','W','S'])
    az.set_yticks([])
    az.tick_params(labelsize=15)

