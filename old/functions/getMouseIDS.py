# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:46:58 2020

@author: kasum
"""

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle

###############################################################################
###Setting Directory and Params
###############################################################################
data_directory   = r'C:\Users\kasum\Dropbox\ADn_Project'
info             = pd.read_excel(os.path.join(data_directory,'experimentsMASTER.xlsx')) #directory to file with all exp data info

strain='wt' #you can equally specify the mouse you want to look at
exp='standard'
cond1='cueA_light'
cond2='cueB_light'
cond3= 180


#################################################################################
###Preselect Rows of Interest for group analysis
#################################################################################

idx=[] #index for all the rows that meet exp condition
idx2=[] #index for all the rows that meet cond1 and 2
for i in range(len(info)):
    if np.any(info.iloc[i,:].str.contains(exp)) and np.any(info.iloc[i,:].str.contains(strain)) : #and np.any(info.iloc[i,:].str.contains(cond1)):
        idx.append(i)
        if (np.any(info.iloc[i,:].str.contains(cond1)==True) and np.any(info.iloc[i,:].str.contains(cond2))) and info.cue_ang[i]==cond3:
            idx2.append(i)
        
All_ids=list(set([info.session[i].split('-')[0] for i in idx2])) 

All_hd_cells=0
for i in idx2:
    if isnan(info.hd_cells[i]):
        ii=i-1
        cell=info.hd_cells[ii]
    else:
        cell=info.hd_cells[i]
    print(cell)
    All_hd_cells+=cell

      
            
print(All_ids)
print(All_hd_cells)