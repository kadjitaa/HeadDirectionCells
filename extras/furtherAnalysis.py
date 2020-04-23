# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:40:38 2019

@author: kasum
"""
import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *



wake_ep_1=nts.IntervalSet(start=wake_ep.loc[0,'start'], end =wake_ep.loc[0,'end'])
wake_ep_2=nts.IntervalSet(start=wake_ep.loc[1,'start'], end =wake_ep.loc[1,'end'])
wake_ep_3=nts.IntervalSet(start=wake_ep.loc[2,'start'], end =wake_ep.loc[2,'end'])
wake_ep_4=nts.IntervalSet(start=wake_ep.loc[3,'start'], end =wake_ep.loc[3,'end'])


i=0

for i in range(len(wake_ep)):
    wake=nts.IntervalSet(start=wake_ep.loc[i,'start'], end =wake_ep.loc[i,'end'])
    print(wake)



 t = np.arange(ep['start'].loc[0], ep['end'].loc[0], duration) #2mins
    t2 = np.hstack((t, ep['end'].loc[0]))
    t3 = np.repeat(t2,2,axis=0)
    t4 = t3[1:-1]
    t5 =t4.reshape(len(t4)//2,2)
    sw_ep=nts.IntervalSet(start=t5[:,0], end =t5[:,1])
    return sw_ep










wake_ep_1=nts.IntervalSet(start=wake_ep.loc[0,'start'], end =wake_ep.loc[0,'start']+ 6e+8)

#sliding window Epochs
t = np.arange(wake_ep_1['start'].loc[0], wake_ep_1['end'].loc[0], 0.6e+8) #2mins
t2 = np.hstack((t, wake_ep_1['end'].loc[0]))
t3 = np.repeat(t2,2,axis=0)
t4 = t3[1:-1]
t5 =t4.reshape(len(t4)//2,2)
wake_sw=nts.IntervalSet(start=t5[:,0], end =t5[:,1])

wake_sw_4=nts.IntervalSet(start=wake_sw.loc[3,'start'], end =wake_sw.loc[3,'end'])

def tc(wake_sw):
    for i in range(len(wake_sw)):
        sw_ep=nts.IntervalSet(start=wake_sw.loc[i,'start'], end =wake_sw.loc[i,'end'])
        sw_tc=computeAngularTuningCurves(spikes,position ['ry'],sw_ep,60)
        
        for j in sw_tc:
        a=b[c]
        print(b1)
        
fig = plt.figure()   



for i in range(1):
        wake_new=nts.IntervalSet(start=wake_sw.loc[i,'start'], end =wake_sw.loc[i,'end'])
        tuning_curve=computeAngularTuningCurves(spikes,position ['ry'],wake_new,60)
        #subplot(projection='polar')
        plot(tuning_curve)
        
        
wake_sw_4=nts.IntervalSet(start=wake_sw.loc[3,'start'], end =wake_sw.loc[3,'end'])
tuning_curve_sw4=computeAngularTuningCurves(spikes,position ['ry'],wake_sw_4,60)
plot(tuning_curve_sw4)

figure()

for i in spikes:
    subplot(4,4,i+1, projection='polar')
    plot(tuning_curve[i],label=str(shank[i]),color='black')