# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:58:48 2020

@author: kasum
"""
def hd_info(tcurve,ep,spikes,position):
    """computes the mutual information for hd cells"""
    I=pd.DataFrame(index=spikes.keys(),columns=['Ispk'])
    for i in spikes.keys():
        lamda_i=tcurve[i].values
        lamda=len(spikes[i].restrict(ep))/ep.tot_length('s')
        
        pos=position['ry'].restrict(ep)
        bins=linspace(0,2*pi,60)
        occu,a=np.histogram(pos, bins)
        occu= occu/sum(occu)
        bits_spk=sum(occu*(lamda_i/lamda)*np.log2(lamda_i/lamda))
        I.loc[i,'Ispk']=bits_spk
    return I      




















