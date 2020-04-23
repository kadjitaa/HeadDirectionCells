# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:00:21 2020

@author: kasum
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


data_files='C:/Users/kasum/Documents/HD_Drift/data'

dark_light=pd.read_hdf(data_files+'/dark_light_dataset.h5')

drift=dark_light.iloc[:,-1][0]

fig=figure(figsize=(11.48,5.59))
gs=GridSpec(3,4)
gs2=GridSpecFromSubplotSpec(3,6, subplot_spec=gs[1:3])
subplot(gs2[0:,0])

l_drft=drift.light.values[drift.light.values <0.01]
scatter(np.random.normal(0,0.09,len(l_drft)),l_drft, s=20,c='white',alpha=0.8, edgecolors='black')

#scatter(0,l_drft.mean(), s=30, c='blue')

d_drft=drift.dark.values[drift.dark.values >0.03]
scatter(np.random.normal(1,0.03,len(d_drft)), d_drft,c='white', s=20,alpha=0.8, edgecolors='black')

#scatter(1,median(d_drft), s=30, c='blue')
plot([0,1],[0.069,0.069],c='k')


gca().set_ylabel('Drift (rad/s)',fontsize=10,labelpad=0.05)
gca().set_xticks([0,1]); gca().set_xticklabels(['Light','Dark'])
#gca().set_yticklabels([-0.25,0,0.25,0.5])
gca().tick_params(labelsize=8.5)
remove_box()
plt.annotate('**',[0.45,0.07], fontsize=7)
gca().spines['bottom'].set_position(('axes',-0.05))
gca().spines['left'].set_position(('axes',-0.05))

wt_val=np.linspace(-0.092, 0.12,len(l_drft)) #wt
zn_val=np.linspace(0.92, 1.12,len(d_drft))

plot(wt_val, np.ones(len(l_drft))*l_drft.mean(), linewidth=3, color='r', alpha=0.9)            
plot(zn_val, np.ones(len(d_drft))*d_drft.mean(), linewidth=3, color='r', alpha=0.9)            


fig_dir='C:/Users/kasum/Dropbox/ADn_Project/paper1_figs'
plt.savefig(fig_dir+'/Fig4.svg',dpi=300, format='svg', bbox_inches="tight", pad_inches=0.05)




