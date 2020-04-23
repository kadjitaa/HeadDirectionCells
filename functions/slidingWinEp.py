# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:51:18 2020

@author: kasum
"""

def slidingWinEp(ep,duration):
    """Generates sliding window epochs with defined durations"""
    
    t = np.arange(ep['start'].loc[0], ep['end'].loc[0], duration) #2mins
    t2 = np.hstack((t, ep['end'].loc[0]))
    t3 = np.repeat(t2,2,axis=0)
    t4 = t3[1:-1]
    t5 =t4.reshape(len(t4)//2,2)
    sw_ep=nts.IntervalSet(start=t5[:,0], end =t5[:,1])
    return sw_ep