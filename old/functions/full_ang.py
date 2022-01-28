# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:35:30 2020

@author: kasum
"""

def full_ang(ep,position):
    'generates a rolling tsd with all the time windows in which all angular bins were sampled.' 
    starts=[]
    ends=[]
    count=np.zeros(60-1)
    bins=linspace(0,2*np.pi,60)
    ang_pos=position.restrict(ep)
    ang_time=ang_pos.times()
    
    idx=np.digitize(ang_pos,bins)-1
    
    start=0
    for i,j in enumerate(idx):
        count[j]+=1
        if np.all(count>=1):
            starts.append(start)
            ends.append(i)
            count=np.zeros(60-1)
            start=i+1
    
    t_start=ang_time[starts]
    t_end=ang_time[ends]
    full_ang_ep=nts.IntervalSet(start=t_start,end=t_end)
    
    return full_ang_ep