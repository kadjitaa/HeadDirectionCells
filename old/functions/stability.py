# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:40:28 2020

@author: kasum
"""

def stability(ep,spikes,position):
    """Computes the spatial correlation of HD cells within a single session
    and returns a correlation value"""
    dur= diff(ep)/2
     #duration must be in microsecs
    ep=slidingWinEp(ep,dur)  
     
    r=pd.DataFrame(index=spikes.keys(), columns=['spatial_corr','pval_corr'])
    ep1=ep.loc[0]
    ep2=ep.loc[1]
    ep1=nts.IntervalSet(ep1.start,ep1.end)
    ep2=nts.IntervalSet(ep2.start,ep2.end)

    tcurve1=computeAngularTuningCurves(spikes,position,ep1,60)
    tcurve2=computeAngularTuningCurves(spikes,position,ep2,60)
    
    for k in spikes.keys():
        r.loc[k,'spatial_corr']=scipy.stats.pearsonr(tcurve1[k].values,tcurve2[k].values)[0]
        r.loc[k,'pval_corr']=scipy.stats.pearsonr(tcurve1[k].values,tcurve2[k].values)[1]
    return r