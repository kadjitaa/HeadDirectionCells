# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:46:30 2020

@author: kasum
"""

def firing_rates(tcurve, ep,spikes): #duration must be in microsecs
#mean firing rate is the total number of emitted spikes divided by the total duration, ignoring occupancy
    f_rates=pd.DataFrame(index=tcurve.columns, columns=['PFD','peak_fr','mean_fr'])
    for k in tcurve.columns:
        pFD=tcurve[k].idxmax(axis=0)
        pFR=tcurve[k].max()
       # mFR=sum(tcurve[k].values)/len(tcurve[k])
        mFR=len(spikes[k].restrict(ep))/ep.tot_length('s')
        f_rates.loc[k,'PFD']=pFD
        f_rates.loc[k,'peak_fr']=pFR
        f_rates.loc[k,'mean_fr']=mFR 
        
    return  f_rates