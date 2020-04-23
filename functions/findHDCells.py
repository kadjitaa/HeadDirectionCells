# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:56:01 2020

@author: Originally written by Guillaume Veijo and modified by kasum to include HD score,etc
"""

def findHDCells(tuning_curves,ep,spikes,position):
    """
        Peak firing rate larger than 1
        and Rayleigh test p<0.001 & z > 100
    """
    cond1 = pd.DataFrame(tuning_curves.max()>1.0)
    angle = position.restrict(ep)
    
    from pycircstat.tests import rayleigh
    stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
    
    for k in tuning_curves:
        stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
        #stat.loc[k]=  rayleigh(position['ry'].restrict(ep).values , position['ry'].realign(spikes[k].restrict(ep)))
                
    
    rMean=pd.DataFrame(index=tuning_curves.columns, columns=['hd_score'])   
    
    for k in tuning_curves:  
        """computes the rayleigh vector length as hdScore. 
        """
        spk = spikes[k]
        spk = spk.restrict(ep)
        angle_spk = angle.realign(spk)
        C = np.sum(np.cos(angle_spk.values))
        S = np.sum(np.sin(angle_spk.values))
        Rmean = np.sqrt(C**2  + S** 2) /len(angle_spk)
        rMean.loc[k]=Rmean
        
    stat['hd_score']=rMean
    
    cond2 = pd.DataFrame(np.logical_and(stat['pval']<0.001,stat['z']>40))
    cond3 = pd.DataFrame(rMean['hd_score']>=0.5)
    tokeep=(cond1[0]==True) & (cond2[0]==True) & (cond3['hd_score']==True)
    stat['hd_cells']=tokeep