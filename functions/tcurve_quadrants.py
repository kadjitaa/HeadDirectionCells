#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:37:39 2020

@author: Mac
"""
##########################################################################
#PARAMETERS
##########################################################################
#Define position as an array
pos_x=array(position['x'].restrict(wake_ep_2))
pos_y=array(position['z'].restrict(wake_ep_2))


x_cen=(pos_x.max()+pos_x.min())/2  
y_cen=(pos_y.max()+pos_y.min())/2


#ZONES
##############################################################################
#upper left
##############################################################################
up_left=(x_cen>pos_x) & (y_cen<pos_y) #defining quadrant
pos_ry_upleft=position['ry'].restrict(wake_ep_2)[up_left]

starts=[]
ends=[]
start=pos_ry_upleft.index[0]

for i in range(len(pos_ry_upleft)-1):
    t=pos_ry_upleft.index[i]
    t1=pos_ry_upleft.index[i+1]
    d=t1-t
    if d>9000:
        starts.append(start)
        ends.append(t)
        start=t1
up_left_eps=pd.DataFrame(data=[starts,ends], index=['start','end']).T 
## clean up to remove extremly small epochs
for i in range(len(up_left_eps)):
    if diff(up_left_eps.loc[i]) == 0:
        up_left_eps=up_left_eps.drop([i])
        
ul_eps=nts.IntervalSet(start=up_left_eps['start'],end=up_left_eps['end'])

tc_ul=computeAngularTuningCurves(spikes,position['ry'],ul_eps,60)


fig=figure()
fig.suptitle('Average Tuning Curve for Session')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tuning_curves_2[i],label=str(i),color='grey', linewidth=3.5)
    ax2.set_xticklabels([])
    legend()


fig=figure()
fig.suptitle('ul')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tc_ul[i],label=str(i),color='r', linewidth=2)
    ax2.set_xticklabels([])
    #legend()


###################################################################################
#upper right
###################################################################################
up_right=(x_cen<pos_x) & (y_cen<pos_y)
pos_ry_upright=position['ry'].restrict(wake_ep_2)[up_right]

starts=[]
ends=[]
start=pos_ry_upright.index[0]

for i in range(len(pos_ry_upright)-1):
    t=pos_ry_upright.index[i]
    t1=pos_ry_upright.index[i+1]
    d=t1-t
    if d>9000:
        starts.append(start)
        ends.append(t)
        start=t1
up_right_eps=pd.DataFrame(data=[starts,ends], index=['start','end']).T 
## clean up to remove extremly small epochs
for i in range(len(up_right_eps)):
    if diff(up_right_eps.loc[i]) == 0:
        up_right_eps=up_right_eps.drop([i])
        
ur_eps=nts.IntervalSet(start=up_right_eps['start'],end=up_right_eps['end'])

tc_ur=computeAngularTuningCurves(spikes,position['ry'],ur_eps,60)

fig=figure()
fig.suptitle('ur')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tc_ur[i],label=str(i),color='k', linewidth=2)
    ax2.set_xticklabels([])
    #legend()


###############################################################################
#bottom left
###############################################################################
b_left=(x_cen>pos_x) & (y_cen>pos_y)
pos_ry_bleft=position['ry'].restrict(wake_ep_2)[b_left]

starts=[]
ends=[]
start=pos_ry_bleft.index[0]

for i in range(len(pos_ry_bleft)-1):
    t=pos_ry_bleft.index[i]
    t1=pos_ry_bleft.index[i+1]
    d=t1-t
    if d>9000:
        starts.append(start)
        ends.append(t)
        start=t1
b_left_eps=pd.DataFrame(data=[starts,ends], index=['start','end']).T 
## clean up to remove extremly small epochs
for i in range(len(b_left_eps)):
    if diff(b_left_eps.loc[i]) == 0:
        b_left_eps=b_left_eps.drop([i])
        
bl_eps=nts.IntervalSet(start=b_left_eps['start'],end=b_left_eps['end'])

tc_bl=computeAngularTuningCurves(spikes,position['ry'],bl_eps,60)

fig=figure()
fig.suptitle('bl')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tc_bl[i],label=str(i),color='magenta', linewidth=2)
    ax2.set_xticklabels([])
    #legend()



################################################################################
# bottom_right
################################################################################
b_right=(x_cen<pos_x) & (y_cen>pos_y)
pos_ry_bright=position['ry'].restrict(wake_ep_2)[b_right]

starts=[]
ends=[]
start=pos_ry_bright.index[0]

for i in range(len(pos_ry_bright)-1):
    t=pos_ry_bright.index[i]
    t1=pos_ry_bright.index[i+1]
    d=t1-t
    if d>9000:
        starts.append(start)
        ends.append(t)
        start=t1
b_right_eps=pd.DataFrame(data=[starts,ends], index=['start','end']).T 
## clean up to remove extremly small epochs
for i in range(len(b_right_eps)):
    if diff(b_right_eps.loc[i]) == 0:
        b_right_eps=b_right_eps.drop([i])
        
br_eps=nts.IntervalSet(start=b_right_eps['start'],end=b_right_eps['end'])

tc_br=computeAngularTuningCurves(spikes,position['ry'],br_eps,60)


fig=figure()
fig.suptitle('br')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tc_br[i],label=str(i),color='blue', linewidth=2)
    ax2.set_xticklabels([])
<<<<<<< Updated upstream
    #legend()
###################################################################################    
    





################################################################################
###PATH PLOTS FOR EACH QUADRANT
################################################################################


figure()
for i in range(len(ul_eps)):
    ep_t=nts.IntervalSet(start=ul_eps.loc[i,'start'], end=ul_eps.loc[i,'end'])
    plot(position['x'].restrict(ep_t),position['z'].restrict(ep_t),c='r')

for i in range(len(ur_eps)):
    ep_t=nts.IntervalSet(start=ur_eps.loc[i,'start'], end=ur_eps.loc[i,'end'])
    plot(position['x'].restrict(ep_t),position['z'].restrict(ep_t),c='k')

for i in range(len(bl_eps)):
    ep_t=nts.IntervalSet(start=bl_eps.loc[i,'start'], end=bl_eps.loc[i,'end'])
    plot(position['x'].restrict(ep_t),position['z'].restrict(ep_t),c='magenta')
    
for i in range(len(br_eps)):
    ep_t=nts.IntervalSet(start=br_eps.loc[i,'start'], end=br_eps.loc[i,'end'])
    plot(position['x'].restrict(ep_t),position['z'].restrict(ep_t),c='blue')
