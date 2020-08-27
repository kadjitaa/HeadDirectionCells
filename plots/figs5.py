# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:34:02 2020

@author: kasum
"""
'''
Computes the frate heatmaps (place-field like) and corresponding tuning curves for two epochs
'''


fig_dir='C:/Users/kasum/Dropbox/ADn_Project/paper1_figs'


gs=GridSpec(9,2)
fig=figure(figsize=(4,12))

eps=[ep1,ep2]
for ep_i,ep in enumerate(eps):
    GF, ext = computePlaceFields(spikes, position[['x', 'z']], ep, 70)
    for i,k in enumerate(spikes.keys()):
       subplot(gs[i,ep_i])
       tmp = gaussian_filter(GF[k].values,sigma = 2.5)
       #for i,v in enumerate(tmp):
           #for j,x in enumerate(tmp):    
               #if tmp[i][j] < 0:
                   #tmp[i][j]=NaN
       im=imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
       cbar=fig.colorbar(im,orientation='vertical')
    #cbar.ax.tick_params(labelsize=12)
       #cbar.ax.set_ylabel('Firing Rate (Hz)',size=8)
      # plt.colorbar(im, cax = fig.add_axes([0.612, 0.535, 0.025, 0.17]))#   left/right  up/down  width height
       #gca().invert_yaxis()
       gca().axis('off')
plt.subplots_adjust(top=0.981, bottom=0.019,left=0.063,right=0.937,hspace=0.188,wspace=0.0)
#plt.savefig(fig_dir+'/Fig5a.svg',dpi=300, format='svg')


gs=GridSpec(9,2)
fig=figure(figsize=(4,12))
tcs=[tcurv_1,tcurv_2]

for l,tc in enumerate(tcs):    
    for i in spikes.keys():
        subplot(gs[i,l], projection='polar')
        plot(tc[i],label=str(i))
        remove_polarAx(gca(),True)
        gca().set_yticks([])

#plt.savefig(fig_dir+'/Fig5b.svg',dpi=300, format='svg')
