# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:50:50 2020

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
import math
import glob, os    



##############################################################################
#READ
##############################################################################
data_files='C:/Users/kasum/Documents/HD_Drift/data'
all_files = glob.glob(os.path.join(data_files, "*.h5"))

#################################################
##PARAMS --open the all_files var and assign the corresponding idx
#################################################
gnat=[3]
rd=[7]
wt=[18]
angles=[180]
loc=[1,2]

fig,ax=subplots(figsize=(3.5,6))
val=7
i=0
for val,ang in zip(wt,angles):
    a=array(pd.DataFrame(pd.read_hdf(all_files[val])['circMean'][0]['wallA']).astype('float'))#.values*(180/np.pi))
    b=array(pd.DataFrame(pd.read_hdf(all_files[val])['circMean'][0]['wallB']).astype('float'))#.values*(180/np.pi))
    
    d=rad2deg(abs(np.arctan2(sin(b-a),cos(b-a))))
    #d=delete(d,-3)
    figure();sns.distplot(d,fit=norm,kde=False)
    gca().set_xlim(0,360)
    plot([180,180],[0,0.09],'--',c='r')
    
        a=array(pd.read_hdf(all_files[val])['circMean'][0]['cueA_light'].values.astype('float'))#.values*(180/np.pi))
        a[0]
 delete(d,-3)

    
    r_diff=sort(ang-(abs(a-b)))
    gnat=ax.scatter(np.random.normal(i,0.08,len(r_diff)), r_diff, s=60,c='w', edgecolors='black')
ax.legend([rd,gnat],['rd1','cg_blind'],loc='upper left',fontsize=11,markerscale=0.5)


i=0
for val,ang in zip(rd,angles):
    a=np.unwrap(pd.read_hdf(all_files[val])['circMean'][0]['cueA_light'].values*(180/np.pi))
    b=np.unwrap(pd.read_hdf(all_files[val])['circMean'][0]['cueB_light'].values*(180/np.pi))
    r_diff=sort(ang-(abs(a-b)))
    rd=ax.scatter(np.random.normal(i,0.08,len(r_diff)), r_diff, s=60,c='grey', edgecolors='black')
    i+=1
ax.legend([wt,rd,gnat],['wt','rd1','cg_blind'],loc='upper left',fontsize=11,markerscale=0.5)

i=0
for val,ang in zip(wt,angles):
    a=np.unwrap(pd.read_hdf(all_files[val])['circMean'][0]['cueA_light'].values*(180/np.pi))
    b=np.unwrap(pd.read_hdf(all_files[val])['circMean'][0]['cueB_light'].values*(180/np.pi))
    r_diff=sort(ang-(abs(a-b)))
    wt=ax.scatter(np.random.normal(i,0.08,len(r_diff)), r_diff, s=60,c='blue', edgecolors='black')
    i+=1
ax.legend([wt,rd,gnat],['wt','rd1','cg_blind'],loc='upper left',fontsize=11,markerscale=0.8)



gca().set_xticks([0,1])
gca().set_xticklabels(['cg_blind','rd1'])
gca().spines['right'].set_visible(False)
gca().spines['top'].set_visible(False)
gca().set_ylabel('HD Shift Difference (Floor rotation ang -'+r' $\Delta$ HD)$^\circ$',size=14)
gca().set_xlabel('90$^\circ$ Floor Rotation',size=14)
gca().set_ylim(-30,180)
gca().tick_params(labelsize=12)


#Extras
gnat_180_wall.to_hdf(data_files+'/gnat_180_wall.h5',mode='a',key='gnat_180_wall') #save file
cueA=np.unwrap(all_pfd['cueA_light'].values *( 180/np.pi))
cueB=np.unwrap(all_pfd['cueB_light'].values *( 180/np.pi))

##############
#regular view
##########
figure();
val=5
for val in range(len(all_files)):
    a=np.unwrap(pd.read_hdf(all_files[val])['pfd'][0].iloc[:,0].values)
    b=np.unwrap(pd.read_hdf(all_files[val])['pfd'][0].iloc[:,1].values)
    scatter(a,b)
    gca().set_xlim(-10,20)
    gca().set_ylim(-10,20)
    
