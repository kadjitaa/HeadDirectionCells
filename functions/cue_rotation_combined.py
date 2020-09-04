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
#data_directory   = r'C:\Users\kasum\Dropbox\ADn_Project' #win
data_directory   = '/Users/Mac/Dropbox/ADn_Project' #Mac

info             = pd.read_excel(os.path.join(data_directory,'experimentsMASTER.xlsx')) #directory to file with all exp data info
contents     = list(info.iloc[i,:])

strain='rd1' #you can equally specify the mouse you want to look at
strain1='gnat'
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
    if (exp in list(info.iloc[i,:])) and (strain in list(info.iloc[i,:]) or strain1 in list(info.iloc[i,:])):
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
all_gtypes=pd.DataFrame(columns=['strain'])
all_stats=[]
###############################################################################
###Data Processing
##############################################################################
plt.figure()
mx_dir='/Volumes/MyBook'
for x,s in enumerate(idx2):
    path=mx_dir+info.dir[s].replace('\\',"/").split(':')[1]   
#Win    
#plt.figure()
#for x,s in enumerate(idx2):
#    path=info.dir[s].replace('\\',"/")
  
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
 
        
    stats=findHDCells(tcurv_2,ep2,spikes,position['ry'],cut_off=0.6)
    hd_cells=stats['hd_cells']==True
    all_stats.append(stats)
    
    circ_mean,_ = computeCircularStats([ep1,ep2],spikes,position['ry'],[cond1,cond2])     
    
    
    cond3=info.rot_ang[s]
    cond4=info.rot_dir[s]
    gtype=info.genotype[s]
    
    gtypes=pd.DataFrame(index=np.arange(len(circ_mean)),columns=['strain'])
    gtypes.iloc[:,0]=gtype
    
    standard_ang=pd.DataFrame(circ_mean.iloc[:,1].values,columns=['observed'])
   
    rot_ang=pd.DataFrame(index=np.arange(len(standard_ang)),columns=['predicted'])#    
    
    #### the if conditions in the for loop below seperates them based on the direction of the cue rotation (clocwise/anticlockwise)
    for i in range(len(circ_mean)):
        if cond4=='anti':
            rot_ang.iloc[i,0]=(abs((circ_mean.iloc[i,1]+deg2rad(cond3))% (2*np.pi)))
            standard_ang.iloc[i,0]=circ_mean.iloc[i,0]
        else:
            rot_ang.iloc[i,0]=(abs((circ_mean.iloc[i,0]+deg2rad(cond3))% (2*np.pi)))
    #plt.figure(); plt.scatter(standard_ang[hd_cells],rot_ang[hd_cells],label=str(s)+cond4);legend()
    
                         
    all_gtypes=all_gtypes.append(gtypes)
    all_standard=all_standard.append(standard_ang[hd_cells]) 
    all_rots=all_rots.append(rot_ang[hd_cells])
    #plt.scatter(standard_ang[hd_cells],rot_ang[hd_cells],label=str(s));legend()
   

print('Total cell count:', len(all_standard))    


########SCATTER PLOT#######################
###########################################
plt.figure(figsize=(2.75,2.4))
if strain != 'wt':
    for i,x in enumerate(all_standard.index):
        if all_gtypes.iloc[i,0]=='rd1':
            defined_color='red'
        else:
            defined_color='green'
        scatter(all_standard.iloc[i,0],all_rots.iloc[i,0],c=defined_color,alpha=0.5,s=10,zorder=3)
else:        
    plt.scatter(all_standard,all_rots, c='b',alpha=0.5,s=10,zorder=3)


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
#remove_box()

plt.plot([0,2*np.pi],[0,2*np.pi], 'r--')
 
x=np.array(all_standard.values).reshape(-1).astype('float')
y=np.array(all_rots.values).reshape(-1).astype('float')
m, b = np.polyfit(x, y, 1)
#linfit=plt.plot(x, m*x + b,color='k',alpha=0.7, linewidth=2)

plt.subplots_adjust(top=0.856,bottom=0.237,left=0.223,right=0.914,hspace=0.2,wspace=0.2)

plt.text(1.0,0.12,'r= '+ str(f'{scipy.stats.spearmanr(x,y)[0]:.3f}'),size=10,fontweight=2,color='b')

plt.savefig(data_directory+'/WT_cueRot.tiff', dpi=600)


