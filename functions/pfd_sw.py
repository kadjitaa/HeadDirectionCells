# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:48:51 2020

@author: kasum
"""

def PFD_sw(ep,spikes,position,dur): #duration must be in microsecs
    """computes the preferred firing direction of HD cells in a sliding window across time"""
    
    sw_ep=slidingWinEp(ep,dur)       
    max_tcurves=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    max_pRate=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    for i in range(len(sw_ep)):
        sw=sw_ep.loc[i]
        sw=nts.IntervalSet(sw.start,sw.end)
        tcurve=computeAngularTuningCurves(spikes,position,sw,60)
        for k in spikes.keys():
            pFD=tcurve[k].idxmax(axis=0)
            pFR=tcurve[k].max()
            max_pRate.iloc[i,k]=pFR
            max_tcurves.iloc[i,k]=pFD
    return max_tcurves, max_pRate