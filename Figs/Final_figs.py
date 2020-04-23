# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:19:20 2019

@author: kasum
"""
######################
#POLAR 
###############

#reduce the width of the bar graph and place it under just 2 polar plots, reduce the wspacing of the first gs
#reduce the size of the polars 


import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

gs = gridspec.GridSpec(2,6)
fig=plt.figure()
#light
for i,x in enumerate(cells):
    ax=subplot(gs[0,i],projection='polar')
    xticks = ax.xaxis.get_major_ticks()
    xticks[1].label.set_visible(False)
    xticks[3].label.set_visible(False)
    xticks[5].label.set_visible(False)
    xticks[7].label.set_visible(False)
    plot(tuning_curves_2[x],color='black', label=str(x))
    ax.fill_between(tuning_curves_2[x].index,tuning_curves_2[x].values,0, color='b')
    plt.subplots_adjust(right=0.85 ,wspace=0.5) #several other parameters that can help you adjust the arrangement
    legend()
    
#Dark    
for i ,x in enumerate(cells):
    ax2=subplot(gs[0,i],projection='polar')
    xticks = ax2.xaxis.get_major_ticks()
    xticks[1].label.set_visible(False)
    xticks[3].label.set_visible(False)
    xticks[5].label.set_visible(False)
    xticks[7].label.set_visible(False)
    plot(tuning_curves_1[x],color='black', label=str(x))
    plt.fill_between(tuning_curves_1.index,tuning_curves_1[x].values,0, color='magenta') #magenta
    plt.subplots_adjust(wspace=0.5) #several other parameters that can help you adjust the arrangement
    legend()

#gs.update(top=0.9,bottom=0.5,hspace=0.4)

#Legend
light_patch = mpatches.Patch(color='cyan', label='Light (10min)')
dark_patch=mpatches.Patch(color='darkgrey', label='Dark (10min)')
plt.legend(handles=[light_patch,dark_patch],loc='top right',bbox_to_anchor=(1.5,1.2),fontsize=18)
plt.show()
fig.suptitle('KA30_190430', fontsize=25)



##########################################################################
#BAR
##########################################################################
light=findHDCells(tuning_curves_2,wake_ep_2,spikes,position)
dark=findHDCells(tuning_curves_1,wake_ep_1,spikes,position)


N = len(hd_cells)
Light =  light.loc[(hd_cells),'HDScore'] #limit indexing to only HD cells

gs2 = gridspec.GridSpec(3, 4)
ax3=fig.add_subplot(subplot(gs2[1,0:3]))
ind = np.arange(N)    # the x locations for the groups
width = 0.40         # the width of the bars
p1 = ax3.bar(ind, Light, width, color='greenyellow',edgecolor='black')


Dark = dark.loc[(hd_cells),'HDScore']
p2 = ax3.bar(ind + width, Dark, width, color='grey',edgecolor='black')

ax3.set_ylabel('Mean vector length (r)')
ax3.set_xticks(ind + width / 2)
ax3.set_xticklabels((hd_cells))
ax3.set_ylim(0,1)
ax3.legend((p1[0], p2[0]), ('Light', 'Dark'))
#ax.autoscale_view()
ax3.set_xlabel("Cell #")
ax3.yaxis.label.set_size(14)
ax3.xaxis.label.set_size(14)
ax3.tick_params(labelsize=12)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
#ax3.plot(np.arange(10),np.ones(10)*0.4,linestyle='--', color='grey')


################################################
#Line Plot of Mean Vector Length (r)
###############################################


data=pd.DataFrame(index=(range(len(hd_cells))), columns=('Light','Dark'))

for i,j in enumerate(light['HDScore']):
    data.loc[i,'Light']=j
data["Light"]=data.loc[(hd_cells),"Light"]
    
    

for i,j in enumerate(dark['HDScore']):
    data.loc[i,'Dark']=j   
data["Dark"]=data.loc[(hd_cells),"Dark"]
data=data.T


means=pd.DataFrame(index=('Light','Dark'), columns=np.arange(1))

dark_mean=data.loc['Dark'].mean(axis=0)
light_mean=data.loc['Light'].mean(axis=0)
means.loc['Dark',0]=dark_mean
means.loc['Light',0]=light_mean
data=np.array(data)

#plots
ax4=figsize=(4.5,5.5)


ax4.plot(data,color='grey',alpha=0.4)
ax4.plot(np.array(means), color='black',linewidth=3)

ax4.set_ylabel('Mean vector length',size=14)
ax4.set_ylim(0,1)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax4.spines['left'].set_position(('axes',-0.03))
ax4.set_xticks([0,1])
ax4.tick_params(labelsize=12)

ax4.set_xticklabels(['Light','Dark'],fontsize=12)
subplots_adjust(left=0.2,bottom=0.1)  #the higher u want it to go the bigger the val for top-several other parameters that can help you adjust the arrangement


#################################################
##CircularMean----doesnt look robust enough!
#################################################

#light
light_peak_ang, light_peak_fr=PFD(wake_ep_2,spikes,position['ry'],3e+8)
light_peak_ang=light_peak_ang.iloc[:,(hd_cells)]
light_peak_fr=light_peak_fr.iloc[:,(hd_cells)]

#ALL Cells
colormap = plt.cm.Paired #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(light_peak_ang.columns))]

#fig=plt.figure()
gs3 = gridspec.GridSpec(3, 9)

#ax.set_title("Stability", y=0.92,x=1.18, size=19)

ax=subplot(gs3[1:2,2:4],projection='polar')    
    
for i in light_peak_ang.columns:
    plot(light_peak_ang[i],light_peak_fr[i],'o',color=colors[i], markersize=20,alpha=0.85)
    
ax.set_xlabel('LIGHT', size=14)          
ax.legend(bbox_to_anchor=(1.25,1), loc="upper left")

        
#Dark
dark_peak_ang, dark_peak_fr=PFD(wake_ep_1,spikes,position['ry'],3e+8)
dark_peak_ang=dark_peak_ang.iloc[:,(hd_cells)]
dark_peak_fr=dark_peak_fr.iloc[:,(hd_cells)]
    
ax1=subplot(gs3[1:2,4:6],projection='polar')    

for i in dark_peak_ang.columns:
    plot(dark_peak_ang[i],dark_peak_fr[i],'o',color=colors[i], markersize=20,alpha=0.85)  
    
ax1.set_xlabel('DARK', size=14)          


##########################################################
#CORRELATION    

gs4 = gridspec.GridSpec(3, 9)

#light
av1=fig.add_subplot(subplot(gs4[2,1:3]))  

#figure()
#for i in light_peak_ang.columns:
#    scatter(light_peak_ang.iloc[0,i],light_peak_ang.iloc[1,i],c=colors[i],s=60,alpha=0.2)
    
    

av1.scatter(light_peak_ang.iloc[0],light_peak_ang.iloc[1],s=120,alpha=0.85)
av1.set_xlim(0,2*np.pi)
av1.set_ylim(0,2*np.pi)
av1.set_xlabel('Angular HD in 1st Half (rad)',size=14 )
av1.set_ylabel('Angular HD in 2nd Half (rad)',size=14 )
av1.tick_params(labelsize=12)

#Dark
av=fig.add_subplot(subplot(gs4[2,3:5]))       
av.scatter(dark_peak_ang.iloc[0],dark_peak_ang.iloc[1],s=120,alpha=0.85)
av.set_xlim(0,2*np.pi)
av.set_ylim(0,2*np.pi)
av.set_xlabel('Angular HD in 1st Half (rad)',size=14 )
av.set_ylabel('Angular HD in 2nd Half (rad)',size=14 )
av.spines['right'].set_visible(True)
av.tick_params(labelsize=12)

######################
#Mean Firing Rates
########################
light=tuning_curves_2
dark=tuning_curves_1

mean_fr=pd.DataFrame(index=range(len(hd_cells)), columns=('Light','Dark'))

for i in spikes.keys():
    mean_fr.loc[i,'Light']=light[i].values.mean()
mean_fr['Light']=mean_fr.loc[(hd_cells),"Light"]  

for i in spikes.keys():
    mean_fr.loc[i,'Dark']=dark[i].values.mean()
mean_fr['Dark']=mean_fr.loc[(hd_cells),"Dark"]  

mean_fr=mean_fr.T


grp_means=pd.DataFrame(index=('Light','Dark'), columns=np.arange(1))

grp_means.loc['Dark',0]=mean_fr.loc['Dark'].mean(axis=0)
grp_means.loc['Light',0]=mean_fr.loc['Light'].mean(axis=0)
mean_fr=np.array(mean_fr)

#plots
av2=fig.add_subplot(subplot(gs3[2,6]), figsize=(4.5,5.5))

av2.plot(mean_fr,color='grey',alpha=0.4)
av2.plot(grp_means, color='black',linewidth=3)

av2.set_ylabel('Mean firing rate (Hz)',size=14)
#av2.set_ylim(0,1)
av2.spines['right'].set_visible(False)
av2.spines['top'].set_visible(False)

av2.spines['left'].set_position(('axes',-0.03))
av2.set_xticks([0,1])
av2.tick_params(labelsize=12)

av2.set_xticklabels(['Light','Dark'],fontsize=12)
#subplots_adjust(left=0.2,bottom=0.1)  #the higher u want it to go the bigger the val for top-several other parameters that can help you adjust the arrangement

###################################################################
#Peak firing Rates
##################################################################
peak_fr=Peak_fr(wake_ep_2,wake_ep_1,spikes,position['ry']) #light,dark
peak_fr=peak_fr.loc[:,(hd_cells)]
#plot
#gs4=gridspec.GridSpec(3,9)
av3=fig.add_subplot(subplot(gs3[2,7]))

av3.plot(peak_fr, color='grey',alpha=0.4)
av3.plot([peak_fr.loc['Light'].mean(axis=0),peak_fr.loc['Dark'].mean(axis=0)], color='black',linewidth=3 )
av3.set_ylabel('Peak firing rate (Hz)',size=14)
#av2.set_ylim(0,1)
av3.spines['right'].set_visible(False)
av3.spines['top'].set_visible(False)

av3.spines['left'].set_position(('axes',-0.03))
av3.set_xticks([0,1])
av3.tick_params(labelsize=12)

#av3.set_xticklabels(['Light','Dark'],fontsize=12)

####################################################################
#Circular Mean & Variance
####################################################################

#c_mean=circular_mean(wake_ep_4,wake_ep_1,spikes,position['ry'])
#c_var=circular_var(wake_ep_4,wake_ep_1,spikes,position['ry'])

#plot
#bar(np.arange(9),c_mean.loc['Dark'],yerr=c_var.loc['Dark'])







#top=0.941,
#bottom=0.05,
#left=0.034,
#right=0.993,
#hspace=0.318,
#wspace=0.7


