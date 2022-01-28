# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:30:42 2020

@author: kasum
"""

def computePlaceInfo(spikes, position, ep, nb_bins = 60, frequency = 120.0):
    Info=pd.DataFrame(index=spikes.keys(),columns=['bits/spk'])
    position_tsd = position.restrict(ep)
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)    
    for n in spikes:
        position_spike = position_tsd.realign(spikes[n].restrict(ep))
        spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
        occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
        mean_spike_count = spike_count/(occupancy+1)
        place_field = mean_spike_count*frequency    
        place_fields = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
        occus= occupancy/sum(occupancy)
        mFR=len(spikes[n].restrict(ep))/ep.tot_length('s')
        info=occus*(place_fields/mFR)*np.log2(place_fields/mFR)
        info=np.array(info)
        info[isnan(info)]=0 
        Info.loc[n]=sum(info)
    return Info