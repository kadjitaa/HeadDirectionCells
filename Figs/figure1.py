# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:13:22 2020

@author: kasum
"""
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
###########################################################################
##Params    
###########################################################################

#light=PFD_Rates(wake_ep_2,spikes,position['ry'], 1e+6)
#dark=PFD_Rates(wake_ep_1,spikes,position['ry'], 1e+6)

dark_light=pd.read_hdf(data_files+'/dark_light_dataset.h5')

drift=dark_light.iloc[:,-1][0]
###########################################################################



neuron=3  

ep=wake_ep_2 #light
tc=tuning_curves_2
cond=light

ep1=wake_ep_1 #dark
tc1=tuning_curves_1
cond1=dark
###########################################################################
    
fig=figure(figsize=(11.48,5.59))

gs=GridSpec(4,8)


#POLAR PLOT
gs1=GridSpecFromSubplotSpec( 1,1,subplot_spec=gs[0,1]) 
ax=subplot(gs1[:], projection='polar')
l_polar=plot(tc[neuron],color='k',linewidth=2.5)
remove_polarAx(gca(),True)
gca().set_yticks([])
#plt.text(-0.02,1,str(round(tc[3].max(),1)))
#subplots_adjust(right=0.85 ,wspace=0.5)

#spks+ pos
gs2=GridSpecFromSubplotSpec( 1, 1,subplot_spec=gs[0,2:5])
ax1=subplot(gs2[0])
ang_spk=position['ry'].realign(spikes[neuron].restrict(ep)).as_units('s')

posx_idx=position['ry'].restrict(ep).as_units('s').index
posx=position['ry'].restrict(ep).as_units('s').values

plot(posx_idx-posx_idx[0], posx, color='grey', alpha=0.5,linewidth=1)

ang=ang_spk.values
spk=ang_spk.index-ang_spk.index[0]

scatter(spk,ang,zorder=3, s=2,c='r', alpha=0.5)
ax1.set_ylim(0,2*np.pi)

tcks=[0,pi,ang.max()]
plt.yticks(tcks)
gca().set_yticklabels([0,'\u03C0',str(2)+'\u03C0'])
ax1.set_ylabel('Angle',size=11, labelpad=-3)
ax1.set_xticklabels([])
ax1.set_xlim(spk.min(),600)
ax1.tick_params(axis='x',direction='out', width=1)
ax1.tick_params(labelsize=9.5)

remove_box()


#locs,labels=xticks()

plt.tight_layout(pad=1, w_pad=3.1, h_pad=1.6) #'''fixes fig layout'''


#drift fit
gs3=GridSpecFromSubplotSpec(1,1, subplot_spec=gs[0,5:7])
ax2=subplot(gs3[0])
idx=cond[1][neuron].values >= tc[neuron].max() * 0.70
x=cond[0][neuron].index.values[idx]-cond[0][neuron].index.values[idx][0]
y=np.unwrap(cond[0][neuron][idx].values)
sp=scatter(x, y, color='red', alpha=0.5)
m, b = np.polyfit(x, y, 1)
linfit=plt.plot(x, m*x + b, color='k',linewidth=2)
ax.tick_params(labelsize=9.5)



#Regression Fit Line
ax2.set_ylabel('Uwrap angle',size=11,labelpad=-3)
ax2.set_ylim(0,12*np.pi)
tcks=[0,6*np.pi,12*pi]
plt.yticks(tcks)
ax2.set_yticklabels([0,str(6)+'\u03C0',str(12)+'\u03C0'])

gca().set_xlim(-22,600)
ax2.tick_params(labelsize=9.5)
ax2.set_xticklabels([])

ax2.tick_params(axis='x',direction='out',labelsize=9.5, width=1)
remove_box()
#Regression Stats
x=array(cond[0].index[idx]).reshape(-1,1)
y=array(np.unwrap(cond[0][neuron][idx].values)).reshape(-1,1)
regr=linear_model.LinearRegression()

regr.fit(x,y)
reg_pred=regr.predict(y)
regr.coef_

#Pearson correlation
scipy.stats.pearsonr(x[:,0],array(y[:,0]))


#coef= 0.0018 rad/sec
################################################################################
######DARK SESSION##############################################################
#POLAR PLOT
gz1=GridSpecFromSubplotSpec( 1,1,subplot_spec=gs[1,1]) 
az=subplot(gz1[0], projection='polar')
plot(tc1[neuron],color='k', linewidth=2.5)
remove_polarAx(gca(),True)
gca().set_yticks([])
#plt.text(-0.02,1,str(round(tc[3].max(),1)))

#spks+ pos
gz2=GridSpecFromSubplotSpec( 1, 1,subplot_spec=gs[1,2:5])
az1=subplot(gz2[0])
ang_spk=position['ry'].realign(spikes[neuron].restrict(ep1)).as_units('s')
ang=ang_spk.values
spk=ang_spk.index-ang_spk.index[0]
scatter(spk,ang,zorder=3, s=2,c='r', alpha=0.5)
posx_idx=position['ry'].restrict(ep1).as_units('s').index
posx=position['ry'].restrict(ep1).as_units('s').values

plot(posx_idx-posx_idx[0], posx, color='grey', alpha=0.5,linewidth=1)
az1.set_ylim(0,2*np.pi)

tcks=[0,pi,2*np.pi]
plt.yticks(tcks)
gca().set_yticklabels([0,'\u03C0',str(2)+'\u03C0'])

az1.set_ylabel('Angle',size=11,labelpad=-3)
az1.set_xlabel('Time (s)',size=11)
az1.tick_params(labelsize=9.5)
az1.set_xlim(spk.min(),600)

remove_box()

#drift fit
gz3=GridSpecFromSubplotSpec(1,1, subplot_spec=gs[1,5:7])
az2=subplot(gz3[0])

idx=cond1[1][neuron].values >= tc1[neuron].max() * 0.70
x=cond1[0][neuron].index.values[idx]-cond1[0][neuron].index.values[idx][0]
y=np.unwrap(cond1[0][neuron][idx].values)
sp=scatter(x, y, c='r', alpha=0.5)
m, b = np.polyfit(x, y, 1)
linfit=plt.plot(x, m*x + b, color='k',linewidth=2)
remove_box()
#Regression Fit Line
az2.tick_params(labelsize=9.5)

az2.set_ylabel('Uwrap angle',size=11,labelpad=-3)

az2.set_ylim(0,12*pi)
tcks=[0, 6*pi, 12*pi]
plt.yticks(tcks)
az2.set_yticklabels([0,str(6)+'\u03C0',str(12)+'\u03C0'])
gca().set_xlabel('Time (s)',size=10)
gca().set_xlim(-22,600)


#Regression Stats
x=array(cond[0].index[idx]).reshape(-1,1)
y=array(np.unwrap(cond[0][neuron][idx].values)).reshape(-1,1)
regr=linear_model.LinearRegression()

regr.fit(x,y)
reg_pred=regr.predict(y)
regr.coef_

#Pearson correlation
scipy.stats.pearsonr(x[:,0],array(y[:,0]))

#coef=0.05 rad/s for dark


legend(['Linear_fit','Spikes'],bbox_to_anchor=(1.05,2.25),fontsize=10)
'''
##########################################################################
###BOXPLOT
###########################################################################
gs7=GridSpecFromSubplotSpec( 6,5,subplot_spec=gs[2:4,1:3]) 
ay=subplot(gs7[1:5,1:-1])

bp=boxplot([drift[drift.light<0.01].light,drift[drift.dark>0.03].dark],showcaps=False,boxprops=dict(linewidth=2))
plot([1,2],[0.069,0.069],'-k', linewidth=0.8)
plt.setp(bp['medians'], color='tomato',linewidth=2); 
gca().set_xticklabels(['Light','Dark'], fontsize=11)
gca().set_ylabel('Drift (rad/s)', fontsize=11)
gca().tick_params(labelsize=12)
remove_box()
#plt.subplots_adjust(left=0.2,right=0.85 ,wspace=1)
#gca().set_ylim(0,0.06)
#gca().spines['left'].set_position(('axes',-0.00)) #
#gca().spines['bottom'].set_position(('axes',-0.05)) #


p_val=print(scipy.stats.mannwhitneyu(drift['light'], drift['dark']))
p_loc=plt.text(1.0,0.070, 'P < 0.0001',size=8)
d_loc=plt.text(-.9,0.120, 'DARK', rotation='vertical', size=16)
l_loc=plt.text(-.9,0.185, 'LIGHT', rotation='vertical',size=16)

a_loc=plt.text(-.9,0.23, '(a)',size=12,fontweight='bold')
b_loc=plt.text(1.30,0.23, '(b)',size=12,fontweight='bold')
c_loc=plt.text(8.64,0.23, '(c)',size=12,fontweight='bold')
d_loc=plt.text(-.9,0.0730, '(d)',size=12,fontweight='bold')


'''
fig_dir='C:/Users/kasum/Dropbox/ADn_Project/paper1_figs'
plt.savefig(fig_dir+'/Fig0.eps',dpi=300, format='eps', bbox_inches="tight", pad_inches=0.05)


sys.exit()



##############################################################################
########################Final modifications##################################
#shared legends

plt.text(-2,90,'(a)',size=14)
l_drift=plt.text(260,2 ,'Drift= 0.0018 rad/s')
#coef= 0.0018 rad/sec
#coef=0.05 rad/s for dark









###############################################################################
###SAVE FIGS##################################################################




