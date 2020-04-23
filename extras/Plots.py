# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:11:24 2019

@author: kasum
"""
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import circmean
###################################################################
#Sliding_Win Polar PLTS
###################################################################
light_ep=slidingWinEp(wake_ep_2,1.6e+7)
#dark_ep=slidingWinEp(wake_ep_1,1.2e+8)

cell_index=0 #Define cell id

for i in spikes.keys():
    for j in range(len(light_ep)):
        new_ep=nts.IntervalSet(start=light_ep.loc[j,'start'], end =light_ep.loc[j,'end'])
        tcurv=computeAngularTuningCurves(spikes,position ['ry'],new_ep,60)
        cell=tcurv[i]  
        ax=subplot(gs[i,j],projection='polar')
        #ax.set_facecolor('yellow')  
        plot(cell, color='black')
        ax.fill_between(tuning_curves_4.index,tcurv[i].values,0, color='greenyellow')







    cell_index=i
    
    






fig = plt.figure(figsize=(8, 10)) 

gs = gridspec.GridSpec(9, 11)

fig=plt.figure()

#ix=[0,1,2,3,4]

#Light
for i in range(len(light_ep)):
    new_ep=nts.IntervalSet(start=light_ep.loc[i,'start'], end =light_ep.loc[i,'end'])
    tcurv=computeAngularTuningCurves(spikes,position ['ry'],new_ep,60)
    cell=tcurv[7]  
    ax=subplot(gs[7,i],projection='polar')
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

 
    
import matplotlib.patches as mpatches

grey_patch = mpatches.Patch(color='lightgrey', label='Dark')
yellow_patch=mpatches.Patch(color='yellow', label='Light')
az=plt.legend(handles=[grey_patch,yellow_patch],loc='upper left')



xticks = ax.xaxis.get_major_ticks()
locs=[1,3,5,7]
for i,x in enumerate(locs):
    xticks[x].label.set_visible(False) #gets rid of the label

ax.yaxis.grid(True) #gets rid of the rings
ax.xaxis.grid(True) #gets rid of the straight lines

#ax.set_yticklabels([]) #gets rid of the firing rate
    
figure()    
ax=subplot(projection='polar')
ax.plot(tuning_curves_4[1])
ax.fill_between(tuning_curves_4[1].index,tuning_curves_4[1].values,0,color='b')

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 9)
subplot(gs[1,1],projection='polar')

plt.fill_between(cell.index,cell.values,0, color='b') #thus fills between a line plot you can add the arg --,where=(x>2) & (x<=3),color='b')




####################################################################
#Preferred Firing Dir
####################################################################
#Define ts containing sliding window epochs of one main Epoch

ep,x=PFD(wake_ep_1,spikes,position['ry'].restrict(wake_ep_1),1.2e+8)
i=6 #Define Cell Id

fig, ax = plt.subplots()

ax.scatter(np.arange(1,(len(ep[i])+1)),ep[i])
ax.set_ylim(0,2*np.pi)
#ax.set_xlim(0,5)
ax.set_ylabel("Change in PFD")
ax.set_xlabel("Minute")
ax.set_yticklabels(['0','60','120','180','240','300','360'])
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(['2','4','6','8','10'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.spines['left'].set_position(('axes',-0.05)) #


figure()
ax=subplot(projection='polar')





#xticks = ax.xaxis.get_major_ticks()
#xticks[8].tick1line.set_visible(False) #gets rid of the tick
#xticks[8].label.set_visible(False) #gets rid of the label

#####################################################################
#Angular HD + Spike Trains
#####################################################################
ep=wake_ep_2 #Define Epoch
tcurve=computeAngularTuningCurves(spikes,position ['ry'],ep,60).idxmax(axis=0)
   
fig, ax=subplots()

ax=subplot(211)
ax.set_ylim(0,2*np.pi)
#ax.set_xticks([])
#ax.set_xticklabels([])
ax.set_ylabel('Angular Head Direction (radians)')
ax.plot(position['ry'].restrict(ep))
ax3=subplot(211)
ax3.plot(decoded_pos)


tcurve=tuning_curves_2_train
ep=wake_ep_2_train
ax2=subplot(212)
ax2.set_ylim(0,2*np.pi)
ax2.set_ylabel('ADn HD Cells')
ax2.set_xlabel('Time (us)')
ax2.set_yticks([])
ax2.set_yticklabels([])
for i,n in enumerate(tcurve):
    print(n)
    #spks=spikes_ka30[i].restrict(ep)
    spks=position['z'].realign(spikes_ka30[i].restrict(ep))
    plot(spks.times(),np.ones(len(spks))*n,'|')




    #plot(spks.times(),np.ones(len(spks))*n,'|') #use this when the cells are tuned



#####################################################################
#Scatter PFD on Polar
#####################################################################
peak_ang, peak_fr=PFD(wake_ep_4,spikes,position['ry'],3e+8)

#ALL Cells

colormap = plt.cm.jet #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(peak_ang.columns))]
fig = plt.figure()
az = fig.add_subplot(111, projection='polar')
for i in peak_ang.columns:
    plot(peak_ang[i],peak_fr[i],'o',color=colors[i], markersize=15)  
    az.legend(loc='upper left')  
    az.legend(bbox_to_anchor=(1.04,1), loc="upper left")



#Individual Cells
cell_id=7
r=peak_fr.iloc[:,cell_id]
theta=peak_ang.iloc[:,cell_id]
area = r.values.astype(int)**2
colors = theta

fig = plt.figure()
az = fig.add_subplot(111, projection='polar')
az.scatter(theta, r, s=area, c=colors, cmap='Paired')







####################################################################
#EXPORTING -Fig & Axis Properties
####################################################################
fig.suptitle("Light Condition: 90 Deg Cue Rotation (A-B-A)", fontsize="25")
fig.legend(('A','B','A'),loc='upper right',fontsize='20')
plt.savefig('KA40-90deg-Light-Rotation-190716.jpg', dpi=400, format='jpeg')

#just in case you just want to use a particular color as your legend


plt.show()
 ###################################################################
'''n = 20    setting a loop to iterate over multiple colors
colors = pl.cm.jet(np.linspace(0,1,n))

for i in range(n):
    pl.plot(x, i*y, color=colors[i])'''
'''
fig, ax=plt.figure()
ax.scatter(list(range(len(wake_pFD_2.iloc[0]))),wake_pFD_2.iloc[0],c=linspace(1,18,18),cmap='Dark2')
'''



######################
#POLAR 
###############

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

gs = gridspec.GridSpec(3, 9)
fig=plt.figure()
for i in spikes:
    ax=subplot(gs[0,i],projection='polar')
    xticks = ax.xaxis.get_major_ticks()
    xticks[1].label.set_visible(False)
    xticks[3].label.set_visible(False)
    xticks[5].label.set_visible(False)
    xticks[7].label.set_visible(False)
    plot(tuning_curves_4[i],color='black', label=str(i))
    ax.fill_between(tuning_curves_4.index,tuning_curves_4[i].values,0, color='greenyellow')
    plt.subplots_adjust(right=0.85 ,wspace=0.5) #several other parameters that can help you adjust the arrangement
    legend()
    
    
for i in spikes:
    ax2=subplot(gs[1,i],projection='polar')
    xticks = ax2.xaxis.get_major_ticks()
    xticks[1].label.set_visible(False)
    xticks[3].label.set_visible(False)
    xticks[5].label.set_visible(False)
    xticks[7].label.set_visible(False)
    plot(tuning_curves_1[i],color='black', label=str(i))
    plt.fill_between(tuning_curves_1.index,tuning_curves_1[i].values,0, color='grey')
    plt.subplots_adjust(wspace=0.5) #several other parameters that can help you adjust the arrangement
    legend()

gs.update(top=0.9,bottom=0.5,hspace=0.4)

#Legend
light_patch = mpatches.Patch(color='greenyellow', label='Light (10min')
dark_patch=mpatches.Patch(color='grey', label='Dark (10min)')
plt.legend(handles=[light_patch,dark_patch],loc='top right',bbox_to_anchor=(2.6,2.9),fontsize=18)

fig.suptitle('KA30_ADn HD Tuning Properties', fontsize=25)



##########################################################################
#BAR
##########################################################################
N = len(spikes.keys())
Light =  light['HDScore']
#LightStd = (20*cm, 30*cm, 32*cm, 10*cm, 20*cm)


#fig, ax = plt.subplots()
gs2 = gridspec.GridSpec(3, 9)
ax3=fig.add_subplot(subplot(gs2[1,0:2]))
ind = np.arange(N)    # the x locations for the groups
width = 0.40         # the width of the bars
p1 = ax3.bar(ind, Light, width, color='greenyellow',edgecolor='black')


Dark = dark['HDScore']
p2 = ax3.bar(ind + width, Dark, width, color='grey',edgecolor='black')

ax3.set_ylabel('Mean vector length (r)')
ax3.set_xticks(ind + width / 2)
ax3.set_xticklabels(('0', '1', '2', '3','4', '5','6','7','8'))
ax3.set_ylim(0,1)
#ax3.legend((p1[0], p2[0]), ('Light', 'Dark'))
#ax.autoscale_view()
ax3.set_xlabel("Cell #")
ax3.yaxis.label.set_size(14)
ax3.xaxis.label.set_size(14)
ax3.tick_params(labelsize=12)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.plot(np.arange(10),np.ones(10)*0.4,linestyle='--', color='grey')

#plt.show()


################################################
#Line Plot of Mean Vector Length (r)
###############################################
_,light=findHDCells(tuning_curves_4,wake_ep_4,spikes,position)
_,dark=findHDCells(tuning_curves_1,wake_ep_1,spikes,position)

data=pd.DataFrame(index=light.index, columns=('Light','Dark'))

for i,j in enumerate(light['HDScore']):
    data.loc[i,'Light']=j
    
for i,j in enumerate(dark['HDScore']):
    data.loc[i,'Dark']=j   
data=data.T


means=pd.DataFrame(index=('Light','Dark'), columns=np.arange(1))

dark_mean=data.loc['Dark'].mean(axis=0)
light_mean=data.loc['Light'].mean(axis=0)
means.loc['Dark',0]=dark_mean
means.loc['Light',0]=light_mean
data=np.array(data)

#plots
ax4=fig.add_subplot(subplot(gs2[2,0:2]), figsize=(4.5,5.5))

ax4.plot(data,color='grey',alpha=0.4)
ax4.plot(np.array(means), color='black',linewidth=3)

ax4.set_ylabel('Mean vector length',size=14)
ax4.set_ylim(0,1)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax4.spines['left'].set_position(('axes',-0.03))
ax4.set_xticks([0,1])
ax4.tick_params(labelsize=12)

ax4.set_xticklabels(['Light','Dark'],fontsize=12)
subplots_adjust(left=0.2,bottom=0.1)  #the higher u want it to go the bigger the val for top-several other parameters that can help you adjust the arrangement


#################################################
##CircularMean----doesnt look robust enough!
#################################################
from scipy.stats import circmean


cir_mean=pd.DataFrame(index=([0,1]),columns=spikes.keys())

dark_sw_ep=slidingWinEp(wake_ep_1,3e+8)

for j in dark_sw_ep.index:
    for i in spikes.keys():
        ep=dark_sw_ep.loc[j]
        ep=nts.IntervalSet(ep.start,ep.end)
        cell_id=position['ry'].realign(spikes[i].restrict(ep))
        cir_mean.loc[j,i]=circmean(cell_id)
       

#light
light_peak_ang, light_peak_fr=PFD(wake_ep_4,spikes,position['ry'],3e+8)

#ALL Cells
colormap = plt.cm.Paired #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(light_peak_ang.columns))]

#fig=plt.figure()
ax=subplot(gs2[1:2,2:4],projection='polar')    
    
for i in light_peak_ang.columns:
    plot(light_peak_ang[i],light_peak_fr[i],'o',color=colors[i], markersize=20,alpha=0.85)  
        
    #ax.legend(loc='upper left')  

#Dark
dark_peak_ang, dark_peak_fr=PFD(wake_ep_1,spikes,position['ry'],3e+8)
    
ax=subplot(gs2[1:2,4:6],projection='polar')    

for i in dark_peak_ang.columns:
    plot(dark_peak_ang[i],dark_peak_fr[i],'o',color=colors[i], markersize=20,alpha=0.85)  
ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
          

##########################################################
#CORRELATION    
#light
fig, ax=subplots()
ax.scatter(light_peak_ang.iloc[0],light_peak_ang.iloc[1])
ax.set_xlim(0,2*np.pi)
ax.set_ylim(0,2*np.pi)
ax.set_xlabel('PFD Angular Head Direction in First Half (rad)' )
ax.set_ylabel('PFD Angular Head Direction in Second Half (rad)' )

#Dark
fig, av=subplots()
av.scatter(dark_peak_ang.iloc[0],dark_peak_ang.iloc[1])
av.set_xlim(0,2*np.pi)
av.set_ylim(0,2*np.pi)
av.set_xlabel('PFD Angular Head Direction in First Half (rad)' )
av.set_ylabel('PFD Angular Head Direction in Second Half (rad)' )
av.spines['right'].set_visible(True)

x=np.array(dark_peak_ang.iloc[0],dtype=float)
y=np.array(dark_peak_ang.iloc[1],dtype=float)
m,b = np.polyfit(x,y, 1) 





##################################################################################################
#SPATIAL CORRELATION HEATMAP
#################################################################################################
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

ep=wake_ep_4 #Define Epoch


r=corr_matrix(ep,spikes,position['ry'])

corr_r=pd.DataFrame(index=spikes.keys(),columns=spikes.keys())
for i in r.columns:
    for j in r.index:
        v=r.loc[i,j][0]
        corr_r.loc[i,j]=abs(v)


new=corr_r.astype('float')

#remove correlations arising as a result of shared tuning
#for i,x in enumerate(new.index):
#   for j in new.columns:
#        if i!=j :
#        #if i!=j and x>=0.5:
#            new.loc[i,j]=0

fig, ax=subplots()
ax1=sns.heatmap(new,vmin=0, vmax=1,cmap='brg',linecolor='grey',linewidths=0.1) #Paired,jet,viridis
ax1.invert_yaxis()
ax1.set_ylabel("Cell #",size=13)
ax1.set_xlabel("Cell #",size=13)
ax1.set_title("Dark: Spatial Correlation Matrix", size=20)               
ax1.tick_params(labelsize=12)
















#import dabest
#from scipy.stats import norm
#two_groups_unpaired=dabest.load(data, idx=("Light","Dark"), paired=True, id_col=data.index)

#dabest.load(data, idx=("Light", "Dark"))

#stat_tc2.to_csv('Dark.csv', encoding='utf-8', index=True)

#dat=pd.read_csv('C:/Users/kasum/Documents/HD_Drift/Light.csv')


n=len(data.columns)
colors = plt.cm.jet(np.linspace(0,1,n))
for i in data.columns:
    plot(data.loc['Dark',1],data.loc['Light',1],linestyle='-')
------------------------------------------------------------------------


