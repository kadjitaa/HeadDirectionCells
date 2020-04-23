# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:41:48 2019

@author: kasum
"""
import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

##################################################################################
##NAV STATS
##################################################################################
#temp ids
wake_ep=wake_ep_1_ka30
position=position_ka30

nav_stats_c=pd.DataFrame(index=np.arange(len(wake_ep)),columns=(['tot_dist','vel']))
nav_stats_w=pd.DataFrame(index=np.arange(len(wake_ep)),columns=(['tot_dist','vel']))
for i in range(len(wake_ep)):
    ep=nts.IntervalSet(start=wake_ep.iloc[i,0],end=wake_ep.iloc[i,1])
    
    pos=pd.DataFrame(index=(range(len(position.restrict(ep)))),columns=['x','z'])
    pos['x']=position['x'].restrict(ep).values
    pos['z']=position['z'].restrict(ep).values
    x_cen=(pos['x'].max()+pos['x'].min())/2
    y_cen=(pos['z'].max()+pos['z'].min())/2
    cen=[x_cen,y_cen]
    
    exp,dist=explore(wake_ep,position)
    
    r=np.sqrt((pos['x']-x_cen)**2+(pos['z']-y_cen)**2) #len of the radius at all points
    cyl_r= r.max() #the radius of the area explored                   meters 56.2cm--cylinder size
    cyl_c=cyl_r-0.10 #2/3 of the cylinder 10cm from per
    
    #Center
    cen=dist[r[0:-1]< cyl_c]
    dist_c=sum(cen)*100
    vels_all_c=(cen*100)*120   #120 is the cam sampling freq, 100 brings the units to cm ####MUST DIVIDE NOT MULTIPLY!
    vel_c=dist_c/(len(cen)/120)
    
    #Wall
    wall=dist[r[0:-1]>=cyl_c]
    dist_w=sum(wall)*100
    vels_all_w=(wall*100)*120  
    vel_w=dist_w/(len(wall)/120)

    #Distance
    nav_stats_c.loc[i,'tot_dist']=dist_c
    nav_stats_w.loc[i,'tot_dist']=dist_w
    #Velocity
    nav_stats_c.loc[i,'vel']=vel_c
    nav_stats_w.loc[i,'vel']=vel_w
            

#occupancy_dark.loc['ka30','tot_dist']=exp['tot_dist'][0]
occu_cg.loc['ka46','tot_vel']=exp['speed'][0] #change to match ep
occu_cg.loc['ka46','c_dist']=nav_stats_c.loc[0,'tot_dist']
occu_cg.loc['ka46','c_vel']=nav_stats_c.loc[0,'vel']
occu_cg.loc['ka46','w_dist']=nav_stats_w.loc[0,'tot_dist']
occu_cg.loc['ka46','w_vel']=nav_stats_w.loc[0,'vel']

#save
occu_light.to_hdf('C:/Users/kasum/Documents/HD_Drift/data'+'/occu_light_dat.h5',mode='a',key='occu_light_dat')
occu_dark.to_hdf('C:/Users/kasum/Documents/HD_Drift/data'+'/occu_dark_dat.h5',mode='a',key='occu_dark_dat')
occu_cg.to_hdf('C:/Users/kasum/Documents/HD_Drift/data'+'/occu_cg_dat.h5',mode='a',key='occu_cg_dat')

#read
occu_dark=pd.read_hdf('C:/Users/kasum/Documents/HD_Drift/data'+'/occu_dark_dat.h5',mode='a',key='occu_dark_dat')
occu_light=pd.read_hdf('C:/Users/kasum/Documents/HD_Drift/data'+'/occu_light_dat.h5',mode='a',key='occu_light_dat')
occu_cg=pd.read_hdf('C:/Users/kasum/Documents/HD_Drift/data'+'/occu_cg_dat.h5',mode='a',key='occu_cg_dat')

####PLOTS     
#fig,ax=subplots();a=bar(np.arange(2),[nav_stats_c.loc[:,'tot_dist'].mean(),nav_stats_w.loc[:,'tot_dist'].mean()],facecolor='None',edgecolor=['r','k'])
#a[1].set_facecolor('r')
#ax.axis('off')
#plt.xticks(np.arange(2), ('inner zone', 'outer zone'))


'''outer zone is defined as the area within 10cm of the cylinder's wall'''
sys.exit()
#############################################################################################
#Plots
#############################################################################################

pos_x_wall=array(pos['x'])[r<cyl_c] #x values that fall within a perimenter of >0.15 
pos_y_wall=array(pos['z'])[r<cyl_c] #y values that fall within a perimenter of >0.15
figure(); plot(position['x'].restrict(ep), position['z'].restrict(ep))
plot(pos_x_wall,pos_y_wall, color='g')

#Velocity Distributions
fig, ax=subplots();center=hist(vels_all_c,bins='auto',color='red');wall=hist(vels_all_w,bins='auto',alpha=0.3) #histogram of the speed distribution
gca().set_ylabel('Counts',size=16)
gca().set_xlabel('Speed (cm/s)',size=16)
ax.legend([center,wall],['inner circle', 'outer circle'])
fig.suptitle('Cg Blind', fontsize=20)
ax.tick_params(labelsize=14)

gca().set_xlim(0,80)
plot(np.ones(2000)*vel_c,np.arange(2000), linestyle='--', color='r')
plot(np.ones(2000)*vel_w,np.arange(2000), linestyle='--', color='b', alpha=0.5)

#dark
vels_c_dark=vels_all_c
vels_c_dark_m=median(vels_c_dark)
vels_w_dark=vels_all_w
vels_w_dark_m=median(vels_w_dark)

#light
vels_c_light=vels_all_c
vels_c_light_m=median(vels_c_light)
vels_w_light=vels_all_w
vels_w_light_m=median(vels_w_light)

#Velocity Distribution 
fig,ax=subplots(figsize=(5.7,7.5));
center=hist(vels_all_w,bins='auto',color='red');wall=hist(vels_all_c,bins='auto',alpha=0.3) #histogram of the speed distribution
gca().set_ylabel('Counts',size=20)
gca().set_xlabel('Speed (cm/s)',size=20)
fig.suptitle('Perimeter Speed', fontsize=20)
legend(['Perimeter', 'Center'],fontsize=16)
ax.tick_params(labelsize=17)
gca().set_xlim(0,60)
gca().set_ylim(0,2500)


plot(np.ones(2000)*median(vels_all_w),np.arange(2000), linestyle='--', color='r') #plots the mean of the distribution, likewise the next line
plot(np.ones(2000)*median(vels_all_c),np.arange(2000), linestyle='--', color='b', alpha=0.5)
plt.subplots_adjust(top=0.935,bottom=0.109,left=0.2,right=0.947,hspace=0.2,wspace=0.2)

#################################################################################################
#Plots Bar with Scatter
#################################################################################################
light_var=blind_dat['hd_info'][blind_dat['hd_cells']]
dark_var=nwhisk_dat['hd_info'][nwhisk_dat['hd_score']>0.5]

fig, ax=subplots(figsize=(4.3,4))
light=ax.bar(0,light_var.median(),color='none', edgecolor='magenta', linewidth=4,width=0.6)
dark=ax.bar(1,dark_var.median(), color='none', edgecolor='brown', linewidth=4, width=0.6)

for i,x in enumerate(light_var):
    l=np.linspace(-0.1, 0.1,len(light_var))
    scatter(l[i],light_var.values[i],s=15,color='black',linewidth=2, zorder=2, alpha=0.8)
    
for i,x in enumerate(dark_var):
    n=np.linspace(0.9, 1.1,len(dark_var))
    scatter(n[i],dark_var.values[i],s=15,color='black',linewidth=2, zorder=2,alpha=0.8)
    
#for i,x in enumerate(blind_var):
#    l=np.linspace(2.4, 2.6,len(blind_var))
#    scatter(l[i],blind_var.values[i],s=15,color='black',linewidth=2,zorder=2,alpha=0.8)
ax.set_ylim(0,1.2)         
ax.set_yticks([0,0.5,1])    
    
ax.set_xticks([0,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticklabels(['Intact whiskers','No whiskers'],size=19)
ax.set_ylabel('Information (bits/sec)',size=17)
ax.tick_params(labelsize=16)

plot(l, np.ones(len(l))*light_var.median(), linewidth=4, color='r', alpha=0.9)            
plot(n, np.ones(len(n))*dark_var.median(), linewidth=4, color='r', alpha=0.9)  

scipy.stats.mannwhitneyu(dark_,new_light)
#plt.subplots_adjust(top=0.963,bottom=0.16,left=0.233,right=0.965,hspace=0.2,wspace=0.2)

#ax.text(0.33,0.009, 'Wildtype', size=16, transform=plt.gcf().transFigure)   

plt.savefig('C:/Users/kasum/Dropbox/ADn_Project/Figs_proposal/New folder/final_figs/'+'vel_p_Light_Dark_cgBlind.svg', dpi=400, format='svg')

##############################################################################################
#Edit might need
##############################################################################################
'''from scipy.ndimage import gaussian_filter
def occu_heatmap(pos):  
    bins=28 
    xpos=pos['X']
    ypos=pos['Y']
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, bins)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, bins)
    occu, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    occu=gaussian_filter(occu,sigma=0.7)
    fig,ax=plt.subplots()
    q=imshow(occu, cmap='jet', interpolation='bilinear')
    ax.axis('off')
    cbar=fig.colorbar(q,orientation='vertical')
    cticks=cbar.ax.get_xticks()
    cbar.set_ticks([])
    return fig



pos=pd.read_csv(r)
'''



