# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:20:48 2020

@author: kasum
"""
def explore(eps, position):
    '''The function takes epochs and position from neuroseries and outputs total distance 
    traveled,distribution of distance traveled per frame and speed''' 
    
    expl=pd.DataFrame(index=range(len(eps)),columns=('tot_dist','speed'))
    for i in range(len(eps)):
        ep=nts.IntervalSet(start=eps.iloc[i,0],end=eps.iloc[i,1])
    
        pos_x=position['x'].restrict(ep)
        pos_y=position['z'].restrict(ep)
        
        x=array(pd.DataFrame(pos_x.values))
        y=array(pd.DataFrame(pos_y.values))
        
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]
        step_size = np.sqrt(dx**2+dy**2)
        #dist = np.concatenate(([0], np.cumsum(step_size)))
        dist=sum(step_size)
        tot_dist=dist*100
        #tot_dist=dist[-1]*100 #converts to cm
        speed=tot_dist/(len(dx)/120) #get the time in seconds using the sampling freq of the camera
        expl.iloc[i,0]=tot_dist
        expl.iloc[i,1]=speed
    return expl,step_size
