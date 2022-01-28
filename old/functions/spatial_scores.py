#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:51:06 2020

@author: Mac
"""
def all_frate_maps(spikes,position,ep):
    
    tms={}
    GF, ext = computePlaceFields(spikes, position[['x', 'z']], ep, 50)
    for i,k in enumerate(spikes.keys()):
       tms[i] = gaussian_filter(GF[k].values,sigma = 2)
    return tms


sw=slidingWinEp(ep1,diff(ep1)//2)
    

sw_ep1=sw.loc[0]; sw_ep1=nts.IntervalSet(sw_ep1.start,sw_ep1.end)
sw_ep2=sw.loc[1]; sw_ep2=nts.IntervalSet(sw_ep2.start,sw_ep2.end)
#tmp,_=frate_maps(spikes,position,sw_ep1)
#tmp1,_=frate_maps(spikes,position,sw_ep2)


tmp=all_frate_maps(spikes,position,ep1)
tmp1=all_frate_maps(spikes,position,sw_ep2)

corr=pd.DataFrame(index=spikes.keys(),columns=['corr'])
for i in range(len(tmp)):
    corr.loc[i]=scipy.stats.pearsonr(tmp[i].flatten(),tmp1[i].flatten())[0]
    print(corr)
