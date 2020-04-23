# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:37:45 2020

@author: kasum
"""

def tc_width(tcurves,spikes):    
    'computes the width of the tuning curve for all cells'
    
    tc_width=pd.DataFrame(index=([0]),columns=spikes.keys())
    for i in tcurves:
        tcurve=tcurves[i]
        max_fr=tcurve.max(axis=0)
        tc_half_w=max_fr/2
    
        tc_max_ang=tcurve.idxmax(axis=0)
        ls_tc=tcurve[tcurve.index < tc_max_ang]
        f1=scipy.interpolate.interp1d(ls_tc.values,ls_tc.index , assume_sorted = False)
        ls=f1(tc_half_w)

        rs_tc=tcurve[tcurve.index > tc_max_ang]
        f2 = scipy.interpolate.interp1d(rs_tc.values,rs_tc.index , assume_sorted = False)
        rs=f2(tc_half_w)
    
        width=rs-ls
        tc_width.loc[0,i]=width
    return tc_width.T