# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:48:10 2019

@author: kasum
"""
#pip install termcolor allows you to color text
import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *
import seaborn as sns
import glob, os    
###################
#import and process csv files
#####################

#Get all csv files in dir
all_files = glob.glob(os.path.join('C:/Users/kasum/Dropbox/bonsai_data/odor_dat', "*.csv")) #make list of paths

#Read and append all csv files
odt_pos=[]   
for file in all_files:
    # Getting the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0]
    # Reading the file content to create a DataFrame
    dfn = pd.read_csv(file)
    # Setting the file name (without extension) as the index name
    dfn.index.name = file_name
    odt_pos.append(dfn)

##odt_pos.append('19090') #Add the date of the experiments if necessary

#Extract all the file names
conds=[]
for i in range(len(odt_pos)):
     cond=odt_pos[i].index.name
     conds.append(cond)



###############################
#Analysis Path Plots
###############################


pos_x=array(pd.DataFrame(odt_pos['X'].dropna()))
pos_y=array(pd.DataFrame(odt_pos['Y'].dropna()))

#Path Plots
mouse=10
mouse1=0  
#Fig 122
fig=subplots(figsize=(14.5,4))
ax=subplot(121)
traj=ax.plot(pos_x,pos_y, color='grey',linewidth=2, alpha=0.7)
ax.set_xlim(16,491)
ax.set_ylim(17,232)
#wall
ax.plot([16,16],[17,232],c='k', linewidth=5)
ax.plot([491,491],[17,232],c='k',linewidth=5)
hor=ax.plot([16,491],[17,17], c='k', linewidth=5);ax.plot([44,560],[238,238],c='k', linewidth=5)
ver=ax.plot([16,491],[232,232], c='k', linewidth=5);ax.plot([560,560],[50,238],c='k', linewidth=5)

avers=ax.scatter(286,130,s=400, facecolors='red',edgecolors='k', linestyle=':',linewidth=3)
avers1=ax.scatter(437,130,s=400, facecolors='red',edgecolors='k', linestyle=':',linewidth=3)

neut=ax.scatter(203,130,s=400, facecolors='white', edgecolors='k', linestyle=':',linewidth=3 )
neut1=ax.scatter(68,130,s=400, facecolors='white', edgecolors='k', linestyle=':',linewidth=3 )

part_up=ax.plot([251,251],[150,230], c='k', linewidth=3)
part_up=ax.plot([251,251],[19,105], c='k', linewidth=3)



ax.set_yticks([])
ax.set_xticks([])
ax.set_title('rd1 Whisker trim + OSN ablated',size=19)




#Fig 121
pos_x=array(pd.DataFrame(odt_pos[mouse1]['X'].dropna()))
pos_y=array(pd.DataFrame(odt_pos[mouse1]['Y'].dropna()))
ax1=subplot(121)
traj=ax1.plot(pos_x,pos_y, color='grey', linewidth=2, alpha=0.7)
ax1.set_xlim(44,560)
ax1.set_ylim(50,238)
ax1.plot([305,305],[50,111],c='k', linewidth=3)
ax1.plot([305,305],[169,240],c='k',linewidth=3)

hor=ax1.plot([44,560],[50,50], c='k', linewidth=5);ax1.plot([44,560],[238,238],c='k', linewidth=5)
ver=ax1.plot([44,44],[50,238], c='k', linewidth=5);ax1.plot([560,560],[50,238],c='k', linewidth=5)
avers=ax1.scatter(270,141,s=400, facecolors='red',edgecolors='k', linestyle=':',linewidth=3)
avers1=ax1.scatter(105,141,s=400, facecolors='red',edgecolors='k', linestyle=':',linewidth=3)

neut=ax1.scatter(336,141,s=400, facecolors='white', edgecolors='k', linestyle=':',linewidth=3 )
neut1=ax1.scatter(511,141,s=400, facecolors='white', edgecolors='k', linestyle=':',linewidth=3 )

ax1.set_yticks([])
ax1.set_xticks([])
ax.legend([traj,neut,avers],['path','Neutral odor','Aversive odor'],loc='upper left',fontsize=11,markerscale=0.5,bbox_to_anchor=(1, 1.03))
ax.set_title('KA56_rd1_OSN ablated',size=19)

plt.suptitle('KA56-200313_rd1_OSN ablated',size=19)

plt.subplots_adjust(top=0.893,bottom=0.049,left=0.014,right=0.99,hspace=0.2,wspace=0.028)
#####################################
#Analysis Stats on Place Preference
#####################################



#index used to seperate chambers
ls_idx=pos_x < 237
rs_idx=pos_x > 270

#left_chamber
l_posx=pos_x[ls_idx]
l_posy=pos_y[ls_idx]
plot(l_posx,l_posy,color='magenta')
#right_chamber
r_posx=pos_x[rs_idx]
r_posy=pos_y[rs_idx]
plot(r_posx,r_posy,color='green')             

#Time computed based on number of frames
freq=30 #camera frame rate
l_tot=len(l_posx)/freq
r_tot=len(r_posx)/freq

#Proportion of time % in each chamber
l_time_s=float(l_tot/(l_tot+r_tot)*100)
r_time_s=float(r_tot/(l_tot+r_tot)*100)

#Determin aversive and neut sides based on start position
if pos_x[0]<237: #started on the left
    d=l_time_s-r_time_s
    neut=l_time_s
    aver=r_time_s
else:
    d=r_time_s-l_time_s
    neut=r_time_s
    aver=l_time_s
    
    


ax1=subplot(133)
ax1.bar([0,0.7],[neut,aver], width=0.5)
ax1.set_xticks([0,0.7])
ax1.set_xticklabels(['neut','avers'],size=13)
ax1.set_ylabel('Time spent (%)', fontsize=18)
ax1.tick_params(labelsize=14)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False) 


#Load data in DataFrame
#odt_disc=pd.DataFrame(columns=['aver','neut','d'])
#odt_disc.loc[conds[mouse],'aver']=aver
#odt_disc.loc[conds[mouse],'neut']=neut
#odt_disc.loc[conds[mouse],'d']=d

#Save data to pc dir
odt_disc.to_hdf('Documents/HD_Drift/data'+'/odt.h5',mode='a',key='odt')

#Read
odt_disc=pd.read_hdf('C:/Users/kasum/Documents/HD_Drift/data'+'/odt.h5',key='odt') 

#Further Data Extraction
wt=odt_disc['d'].iloc[:5].values
zn=odt_disc['d'].iloc[5:].values
eg=pd.DataFrame(index=range(len(odt_disc)),columns=['condition','d'])
eg.iloc[:5,0]='wtype'
eg.iloc[:5,1]=wt
eg.iloc[5:,0]='zinc'
eg.iloc[5:,1]=zn

#Scatter plots + Stats
#fig, ax = plt.subplots()
fig,ax3=subplots(figsize=(3,4))
ax3.scatter(np.ones(len(zn))*2,zn, facecolors='none', edgecolors='k',s=100,linewidths=3,alpha=0.5)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

for i,x in enumerate(wt):
    wt_val=np.linspace(0.9, 1.1,len(wt)) #wt
    zn_val=np.linspace(1.9, 2.1,len(wt))
    scatter(wt_val[i],wt[i], facecolors='none', edgecolors='k', s=100,linewidths=3, alpha=0.5)
    
#mean
plot(wt_val, np.ones(len(wt))*wt.mean(), linewidth=4, color='k', alpha=0.9)            
plot(zn_val, np.ones(len(wt))*zn.mean(), linewidth=4, color='k', alpha=0.9)            
#params
ax3.set_xticks([1,2]) 
ax3.set_yticks([-100,0,100])   
ax3.set_yticklabels([-1,0,1])
ax3.set_ylim(-100,120)
ax3.set_xticklabels(['Wildtype','OSN ablated'],size=17)
ax3.set_ylabel('Place preference\n(time in neutral room - aversive room)',size=13)
ax3.tick_params(labelsize=14)

#stats linking bar
col='k'
x=[1,1]; x2=[1,2]; x3=[2,2]
y=[110,115]; y2=[115,115]; y3=[110,115]
plot(x,y,c=col); plot(x2,y2,c=col); plot(x3,y3,c=col)

#stats
stat,p=scipy.stats.mannwhitneyu(wt,zn)

plt.text(1.2, 117.5, "P = 0.003", size=12)

#######EXTRA CODE##########
'''
filename='/A2929-wt-treated.csv'
dirs=r'F:\EphysData\Experiments\TrackingVids\odor_tracking\csv_tracking'
data=pd.read_csv(dirs+filename, index_col=False,sep=' ')
#figure()
subplot(211)
plot(data['X'],data['Y'])
gca().set_yticks([])
gca().set_xticks([])
plt.title(filename)
scatter(data.iloc[0,0], data.iloc[0,1],s=90,c='r')
'''