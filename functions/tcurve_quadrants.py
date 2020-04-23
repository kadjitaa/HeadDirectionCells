#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:37:39 2020

@author: Mac
"""

#Define position as an array
pos_x=array(position['x'].restrict(wake_ep_2)); pos_xt=array(position['x'].index.values)
pos_y=array(position['z'].restrict(wake_ep_2)); pos_yt=array(position['x'].index.values)


#you may have to manually define the center of your environment for data with irregular path plots
x_cen=(pos_x.max()+pos_x.min())/2  
y_cen=(pos_y.max()+pos_y.min())/2
#c_vert=plot([x_cen,x_cen], [pos_y.min(),pos_y.max()]) #plots vertical line through center for verification
#c_hor=plot([pos_x.min(),pos_x.max()],[y_cen,y_cen])   #plots horizontal line through center for verification


#ZONES
#upper left
up_left=(x_cen>pos_x) & (y_cen<pos_y) #defining quadrant
pos_ry_upleft=position['ry'].restrict(wake_ep_2)[up_left]

#spk_pos_align=pos_ry_upleft.realign(spikes[1].restrict(wake_ep_2))
#scatter(spk_pos_align.index.values, np.ones(len(spk_pos_align))*2.36,marker="|", c='g');plot(pos_ry_upleft)


ul_ep1=nts.IntervalSet(start=2470136649,end=2526800000)
tuning_curves_2_ul=computeAngularTuningCurves(spikes,position['ry'],ul_ep1,60)


figure()
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tuning_curves_2[i],label=str(i),color='r', linewidth=2)
    ax2.set_xticklabels([])
    legend()


fig=figure()
fig.suptitle('ul')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tuning_curves_2_ul[i],label=str(i),color='r', linewidth=2)
    ax2.set_xticklabels([])
    legend()


#upper right
up_right=(x_cen<pos_x) & (y_cen<pos_y)
pos_ry_upright=position['ry'].restrict(wake_ep_2)[up_right]

ur_ep1=nts.IntervalSet(start=2.21651e+09, end=2.26297e+09)
tuning_curves_2_ur=computeAngularTuningCurves(spikes,position['ry'],ur_ep1,60)
fig=figure()
fig.suptitle('ur')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tuning_curves_2_ur[i],label=str(i),color='r', linewidth=2)
    ax2.set_xticklabels([])
    legend()



#bottom left
b_left=(x_cen>pos_x) & (y_cen>pos_y)
pos_ry_bleft=position['ry'].restrict(wake_ep_2)[b_left]
figure();scatter(pos_ry_bleft.index.values,pos_ry_bleft.values, alpha=0.5)
bl_ep1=nts.IntervalSet(start=2.52671e+09, end=2.548e+09)
tuning_curves_2_bl=computeAngularTuningCurves(spikes,position['ry'],bl_ep1,60)
fig=figure()
fig.suptitle('bl')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tuning_curves_2_bl[i],label=str(i),color='r', linewidth=2)
    ax2.set_xticklabels([])
    legend()




# bottom_right
b_right=(x_cen<pos_x) & (y_cen>pos_y)
pos_ry_bright=position['ry'].restrict(wake_ep_2)[b_right]
br_ep1=nts.IntervalSet(start=2.5275e+09, end=2.54789e+09)
tuning_curves_2_bl=computeAngularTuningCurves(spikes,position['ry'],br_ep1,60)
fig=figure()
fig.suptitle('br')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tuning_curves_2_bl[i],label=str(i),color='r', linewidth=2)
    ax2.set_xticklabels([])
    legend()