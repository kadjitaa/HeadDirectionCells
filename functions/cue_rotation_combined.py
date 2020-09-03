# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:04:03 2020

@author: kasum
"""

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from sklearn import datasets, linear_model


###############################################################################
###Setting Directory and Params
###############################################################################
data_directory   = r'C:\Users\kasum\Dropbox\ADn_Project'
info             = pd.read_excel(os.path.join(data_directory,'experimentsMASTER.xlsx')) #directory to file with all exp data info

strain='rd1' #you can equally specify the mouse you want to look at
exp='standard'
cond1='floorA'
cond2='floorB'
#CHECK LINE 85


#################################################################################
###Preselect Rows of Interest for group analysis
#################################################################################

idx=[] #index for all the rows that meet exp condition
idx2=[] #index for all the rows that meet cond1 and 2
for i in range(len(info)):
    if np.any(info.iloc[i,:].str.contains(exp)) and np.any(info.iloc[i,:].str.contains(strain)) : #and np.any(info.iloc[i,:].str.contains(cond1)):
        idx.append(i)
        if (cond1 in list(info.iloc[i,:])) and (cond2 in list(info.iloc[i,:])):
            idx2.append(i)


##############################################################################
###Within Animal Analysis
################################################################################
###Combined Datasets      

all_circMean=pd.DataFrame(columns=([cond1,cond2]))
all_standard=pd.DataFrame(columns=(['observed']))
all_rots=pd.DataFrame(columns=(['predicted']))

all_stats=[]
###############################################################################
###Data Processing
##############################################################################
plt.figure()
for x,s in enumerate(idx2):
    path=info.dir[s].replace('\\',"/")
  
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    episodes = info.filter(like='T').loc[s]
    events  = list(np.where((episodes == cond1) | (episodes== cond2))[0].astype('str'))
    
    spikes, shank                       = loadSpikeData(path)
    #n_channels, fs, shank_to_channel   = loadXML(path)
    position                            = loadPosition(path, events, episodes)
    wake_ep                             = loadEpoch(path, 'wake', episodes)
    
    ep1=nts.IntervalSet(start=wake_ep.loc[int(events[0])-1,'start'], end =wake_ep.loc[int(events[0])-1,'start']+6e+8)
    ep2=nts.IntervalSet(start=wake_ep.loc[int(events[-1])-1,'start'], end =wake_ep.loc[int(events[-1])-1,'start']+6e+8)
        
    tcurv_1 = computeAngularTuningCurves(spikes,position['ry'],ep1,60)
    tcurv_2 = computeAngularTuningCurves(spikes,position['ry'],ep2,60)

    stats=findHDCells(tcurv_2,ep2,spikes,position['ry'],cut_off=0.49)
    hd_cells=stats['hd_cells']==True
    all_stats.append(stats)
    
    circ_mean,_ = computeCircularStats([ep1,ep2],spikes,position['ry'],['cueA_light','cueB_light']) 
    
    cond3=info.floor_ang[s]

    standard_ang=pd.DataFrame(circ_mean.iloc[:,0].values,columns=['observed'])

    rot_ang=pd.DataFrame(index=np.arange(len(standard_ang)),columns=['predicted'])#                 (index=list(np.arange(len(standard_ang))),columns=['New'])
    for i in range(len(circ_mean)):
        rot_ang.iloc[i,0]=((((abs(circ_mean.iloc[i,1]-circ_mean.iloc[i,0]))))-deg2rad(cond3)+ (circ_mean.iloc[i,0]))
    
    plt.scatter(standard_ang,rot_ang,label=str(s))
    legend()
    '''
    all_standard=all_standard.append(standard_ang[hd_cells]) 
    all_rots=all_rots.append(rot_ang[hd_cells])


    
    plt.figure()
    for i,x in enumerate(list(np.where(hd_cells==True)[0])):#spikes.keys():
        subplot(5,8,i+1, projection='polar')
        plot(tcurv_1[x])
        plot(tcurv_2[x])
        remove_polarAx(gca())        
    '''
sys.exit()
print('Total cell count:', len(all_standard))    
########SCATTER PLOT#######################
plt.figure()

plt.scatter(all_standard,all_rots, c='k',s=10,zorder=3)

plt.title('Cue Card Rotation')

gca().set_ylabel('Observed Mean PFD (rad)')
gca().set_xlabel('Predicted Mean PFD (rad)')

gca().set_ylim(0,2*np.pi)
gca().set_xlim(0,2*np.pi)


tcks=[0,pi,2*np.pi]

plt.xticks(tcks)
plt.yticks(tcks)
gca().set_yticklabels([0,'\u03C0',str(2)+'\u03C0'])
gca().set_xticklabels([0,'\u03C0',str(2)+'\u03C0'])
remove_box()



plt.plot([0,2*np.pi],[0,2*np.pi], 'r--')
 
x=np.array(all_standard.values).reshape(-1).astype('float')
y=np.array(all_rots.values).reshape(-1).astype('float')
m, b = np.polyfit(x, y, 1)
linfit=plt.plot(x, m*x + b,color='k',alpha=0.6, linewidth=2)


correlation_matrix = np.corrcoef(x, y)
correlation_xy = correlation_matrix[0,1]
r_squared = print(correlation_xy**2)


plt.text(2.8,5,'R$\mathregular{^2}$='+str((f'{m:.3f}')), size=14,color='red')
