# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:39:02 2019

@author: kasum
"""
import pandas as pd
import neuroseries as nts
from wrappers import *
import os, sys
import matplotlib.pyplot as plt
from pylab import *

from functions import *
import numpy as np

########################################################################
#LINE PLOT WITH SHADDED 
########################################################################
#Params
y=np.random.rand(10)  #numpy array of means
x=np.arange(1,len(y)+1)
sd=np.ones(len(x))*0.15 #numpy array of Stds or Std.Err
c='blueviolet'
shade=0.2  #transparency shade
width= 5 #width of line

#Plot

err1=[]
err2=[]
for i in np.arange(len(y)):
    err1.append(y[i]-sd[i])
    err2.append(y[i]+sd[i])
    
fig, ax= plt.subplots()
ax=subplot(211)
ax.plot(x,y,color=c, linewidth=width)
plt.fill_between(x,err1,err2,color=c, alpha=shade)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_position(('axes',-0.05))
ax.set_xlim(1, len(x))
ax.set_ylim(0,max(err2))
ax.set_xticks(np.arange(1,(len(y))+1))

ax.set_ylabel("Performance (%)")
ax.set_xlabel("CueSF")

fig.legend(('Mean','SD'))

#plot for p val


#ax2=subplot(212)
y1=np.random.rand(10)
#ax2 = ax.twinx()
ax2=subplot(212)
ax2.plot(x,y1)

ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax2.spines['right'].set_position(('data',10.5))
ax2.set_ylim(0,4)
ax2.set_xticks([])
ax2.set_yticks([0,0.5,1]) #set this to reflect the range of vals you want to see
ax2.set_xlim(1, len(x))
ax2.set_ylabel('log(p)')

ax2.yaxis.set_label_coords(-0.065,0.15)

plt.fill_between(x,y1,1,color=c, alpha=shade)

plt.subplots_adjust(wspace=None, hspace=-0.5) #several other parameters that can help you adjust the arrangement
ax2.patch.set_visible(False) #this helps you to hide the extra fig portion that interferes


fig.show()


#yticks = ax2.yaxis.get_major_ticks()
#yticks[2].tick1line.set_visible(True) #gets rid of the tick
#yticks[2].label1.set_visible(True) #gets rid of the label


alpha=np.ones(len(x))*0.75
alpha_line=plot(x,alpha, linestyle='--')
txt=plt.text(10.1, 0.72, r"$\alpha$")


