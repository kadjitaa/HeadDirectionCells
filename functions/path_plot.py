# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:26:05 2020

@author: kasum
"""

def path_plot(eps,position):
     '''The function takes epochs and position from neuroseries and outputs path plots for each epoch'''
     fig=figure()
     for i in range(len(wake_ep)):
        if len(wake_ep)==1:
            ax=subplot()
        else:    
            ax=subplot(2,len(wake_ep),i+1)
        ep=eps.iloc[i]
        ep=nts.IntervalSet(ep[0],ep[1])
        plot(position['x'].restrict(ep),position['z'].restrict(ep),color='red',label=str(i), alpha=0.5) 
        legend()