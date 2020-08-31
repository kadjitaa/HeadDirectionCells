# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:19:09 2020

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
data_directory   = '/Users/Mac/Dropbox/ADn_Project'
info             = pd.read_excel(os.path.join(data_directory,'experimentsMASTER.xlsx')) #directory to file with all exp data info

strain='wt' #you can equally specify the mouse you want to look at
exp='standard'
cond1='cueA_light'
cond2='cueB_light'
cond3= 90


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

##############################################################################
###Within Animal Analysis
################################################################################
###Combined Datasets      
#all_means=pd.DataFrame(columns=([cond1,cond2]))
#all_peaks=pd.DataFrame(columns=([cond1,cond2]))
all_pfd=pd.DataFrame(columns=([cond1,cond2]))
#all_info=pd.DataFrame(columns=([cond1,cond2]))
#all_vLength=pd.DataFrame(columns=([cond1,cond2]))
#all_stability=pd.DataFrame(columns=([cond1,cond2]))
all_circMean=pd.DataFrame(columns=([cond1,cond2]))
#all_circVar=pd.DataFrame(columns=([cond1,cond2]))

#all_light_tc=[]
#all_dark_tc=[]

#gcorr_envA=pd.DataFrame(columns=['EnvB'])
###############################################################################
###Data Processing
##############################################################################
mx_dir='/Volumes/MyBook'
for x,s in enumerate(idx2):
    path=mx_dir+info.dir[s].replace('\\',"/").split(':')[1]
  
    ############################################################################################### 
    # LOADING DATA
    ###############################################################################################
    episodes = info.filter(like='T').loc[s]
    events  = list(np.where((episodes == cond1) | (episodes== cond2))[0].astype('str'))
    
    spikes, shank                       = loadSpikeData(path)
    #n_channels, fs, shank_to_channel   = loadXML(path)
    position                            = loadPosition(path, events, episodes)
    wake_ep                             = loadEpoch(path, 'wake', episodes)
    #sleep_ep                            =loadEpoch(path,'sleep',episodes)
    
    ep1=nts.IntervalSet(start=wake_ep.loc[int(events[0])-1,'start'], end =wake_ep.loc[int(events[0])-1,'start']+6e+8)
    ep2=nts.IntervalSet(start=wake_ep.loc[int(events[-1])-1,'start'], end =wake_ep.loc[int(events[-1])-1,'start']+6e+8)
    #ep_train=nts.IntervalSet(start=wake_ep.loc[int(events[-1])-1,'start'], end =wake_ep.loc[int(events[-1])-1,'start']+3e+8)
   
        
    tcurv_1 = computeAngularTuningCurves(spikes,position['ry'],ep1,60)
    tcurv_2 = computeAngularTuningCurves(spikes,position['ry'],ep2,60)
    #tc_train= computeAngularTuningCurves(spikes,position['ry'],ep_train,60)

    figure(); plot(position['x'].restrict(ep2), position['z'].restrict(ep2), label=str(s))
    legend()
    
    
    stats=findHDCells(tcurv_2,ep2,spikes,position['ry'])
    hd_cells=stats['hd_cells']==True
    
    circ_mean,_              = computeCircularStats([ep1,ep2],spikes,position['ry'],[cond1,cond2])
    _,_, pfd  = computeFiringRates(spikes,[ep1, ep2],[cond1,cond2],[tcurv_1,tcurv_2]) 

    all_circMean=all_circMean.append(circ_mean[hd_cells])           #Circ Mean

    all_pfd=all_pfd.append(pfd[hd_cells])                           #Preferred Firing Dir

    plt.figure()
    for i,x in enumerate(list(np.where(hd_cells==True)[0])):#spikes.keys():
        subplot(5,5,i+1, projection='polar')
        plot(tcurv_1[x])
        plot(tcurv_2[x])
        remove_polarAx(gca())
    
    
    
    
    
    
    
    
    
    
    
    
    sw=slidingWinEp(ep2,diff(ep2)//2)
    

    sw_ep1=sw.loc[0]; sw_ep1=nts.IntervalSet(sw_ep1.start,sw_ep1.end)
    sw_ep2=sw.loc[1]; sw_ep2=nts.IntervalSet(sw_ep2.start,sw_ep2.end)
    #tmp,_=frate_maps(spikes,position,sw_ep1)
    #tmp1,_=frate_maps(spikes,position,sw_ep2)
    frate_maps(spikes,position, ep2)
    path_spk_plot(ep2,spikes,position)
    
    tmp=all_frate_maps(spikes,position,sw_ep1)
    tmp1=all_frate_maps(spikes,position,sw_ep2)
    
    pearson_c=pd.DataFrame(index=spikes.keys(),columns=['EnvB'])
    for i in range(len(tmp)):
        pearson_c.loc[i]=scipy.stats.pearsonr(tmp[i].flatten(),tmp1[i].flatten())[0]
        
    gcorr_envA=gcorr_envA.append(pearson_c)

figure(); boxplot([hd_inf.iloc[:,0].values,place_inf.iloc[:,0].values])
gca().set_xticklabels(['HD','Location'])
gca().set_ylabel('Information (bits/spk)')
    
scipy.stats.mannwhitneyu(hd_inf.values,place_inf.values)   
    
    
    
    
    
corr_envA['blind']=gcorr_envA.values
    
    
    
   # path_plot(ep1,position)

    #makeRingManifold(spikes,ep1,position['ry'],100)    
    #plt.savefig(r'C:\Users\kasum\Dropbox\ADn_Project\200321\rd1_OSN_Iso100_D'+str(x)+'.svg',dpi=900, format='svg', bbox_inches="tight", pad_inches=0.05)

    
    
    
    stats=findHDCells(tcurv_2,ep2,spikes,position['ry'])
    hd_cells=stats['hd_cells']==True
    all_light_tc.append(tcurv_2.loc[:,hd_cells])
    all_dark_tc.append(tcurv_1.loc[:,hd_cells])
    
    
    
    
fig,ax=subplots(figsize=(2.9,2.5))
for i,x in enumerate(range(5,9)):
    ax=subplot(2,2,i+1, projection='polar')
    plot(tcurv_1[x],label=str(x),color='k', linewidth=1.5)
    ax.fill_between(tcurv_1[x].index,tcurv_1[x].values,0, zorder=2,color='grey')
    remove_polarAx(ax,True)
    gca().set_yticks([])
    cell1=plt.text(0.435, 0.79, str(round(tcurv_1[5].max(),1))+'Hz', fontsize=7, transform=plt.gcf().transFigure)
    cell2=plt.text(0.855, 0.79, str(round(tcurv_1[6].max(),1))+'Hz', fontsize=7, transform=plt.gcf().transFigure)
    cell3=plt.text(0.435, 0.37, str(round(tcurv_1[7].max(),1))+'Hz', fontsize=7, transform=plt.gcf().transFigure)
    cell4=plt.text(0.855, 0.37, str(round(tcurv_1[8].max(),1))+'Hz', fontsize=7, transform=plt.gcf().transFigure)
    
    
    

plt.savefig(r'C:\Users\kasum\Dropbox\ADn_Project\200321\rd1_day4iso_OSN.svg',dpi=900, format='svg', bbox_inches="tight", pad_inches=0.05)

    
figure()
for i in spikes.keys():
    rws=int(len(spikes.keys())/4)+1
    ax=subplot(rws,4,i+1, projection='polar')
    plot(tcurv_2[i],label=str(i))
    legend()
    
figure(); plot(position['x'].restrict(ep2), position['z'].restrict(ep2), label=str(s))
legend()
    
    
    
    
    
    
all_dark_tc=pd.concat([all_dark_tc[0],all_dark_tc[1]], axis=1,ignore_index=True )
all_light_tc=pd.concat([all_light_tc[0],all_light_tc[1]], axis=1, ignore_index=True) 
    
    
    
    if x==0:
        all_light_tc=all_light_tc.append(tcurv_2[hd_cells])
        sz=tcurv_2.columns[-1]
    else:
        loc=spikes.keys()
        all_light_tc[locs[hd_cells]]=tcurv_2[hd_cells]
    
    
                                     
    all_tc[all_tc.columns+shape(tuning_curves_2)[1]]=tuning_curves_2
    
    
##QWIK PLOTS########################
path_plot(ep1,position)

sz=(int(len(spikes.keys()))/4)+1
for i in range(len(wake_ep)):
    ep=nts.IntervalSet(start=wake_ep.loc[i,'start'], end=wake_ep.loc[i,'start']+6e8)
    tc=computeAngularTuningCurves(spikes,position['ry'],ep,60)
    figure()
    for x in spikes.keys():
        subplot(sz,4,1+x, projection='polar')
        plot(tc[x])
#########################################################################################
    
    
 
    
    
    
    
    
    
    
    ############################################################################################### 
    # FIRING RATE ANALYSIS
    ###############################################################################################
    stats=findHDCells(tcurv_2,ep2,spikes,position['ry'])
    hd_cells=stats['hd_cells']==True
    mean_frate,peak_frate, pfd  = computeFiringRates(spikes,[ep1, ep2],[cond1,cond2],[tcurv_1,tcurv_2]) 
    all_pfd=all_pfd.append(pfd[hd_cells]) 
    
    
    cueA_unwrap=unwrap(all_pfd['cueA_light'].values)
    cueB_unwrap=unwrap(all_pfd['cueB_light'].values)
    scatter(cueA_unwrap,cueB_unwrap)        
    
    
    
    
    
    mean_vLength                    = computeVectorLength(spikes,[ep1,ep2], position['ry'],[cond1,cond2])
    spatial_corr                    = computeStability([ep1,ep2],spikes,position['ry'],[cond1,cond2])
    
    
    
    figure()
    for i in spikes.keys():
        rws=int(len(spikes.keys())/4)+1
        ax=subplot(rws,4,i+1, projection='polar')
        plot(tcurv_2[i])
    
    #Stats
    
    all_stability=all_stability.append(spatial_corr)      #Spatial Correlation Pearson(r)

    
    
    
    all_pfd=all_pfd.append(pfd[hd_cells])   
    all_means=all_means.append(mean_frate)                #Mean Firing Rates
    all_vLength=all_vLength.append(mean_vLength)  
        

    
    all_means=all_means.append(mean_frate[hd_cells])                #Mean Firing Rates
    all_peaks=all_peaks.append(peak_frate[hd_cells])                #Peak Firing Rates
    all_pfd=all_pfd.append(pfd[hd_cells])                           #Preferred Firing Dir
    all_light_tcurv.append(tcurv_1)
    all_dark_tcurv.append(tcurv_2[hd_cells])
    all_vLength=all_vLength.append(mean_vLength[hd_cells])          #Rayleigh Vector Length
    all_stability=all_stability.append(spatial_corr[hd_cells])      #Spatial Correlation Pearson(r)
    all_circMean=all_circMean.append(circ_mean[hd_cells])           #Circ Mean
    all_circVar=all_circVar.append(circ_var[hd_cells])              #Circ Variance
    all_info=all_info.append(info_hd[hd_cells])                     #Mutual Information
    
    
    
    
    
   

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

fig=figure();

gs = gridspec.GridSpec(2,2)
gs1=  GridSpecFromSubplotSpec(2,3, subplot_spec=gs[0]) 
for i,x in enumerate(cells):
    ax=subplot(gs1[i], projection='polar')
    if i==1:
        ax.set_title('KA55-200305_Dark_90deg Cue Rotation',fontsize=20)
    a=plot(tcurv_1[x])
    remove_polarAx(gca(),True)
    gca().set_yticks([])
    
for i,x in enumerate(cells):
    ax=subplot(gs1[i], projection='polar')
    b=plot(tcurv_2[x])
    remove_polarAx(gca(),True)
    gca().set_yticks([])
        


gs2=  GridSpecFromSubplotSpec(2,4, subplot_spec=gs[1]) 
for i,x in enumerate(cells1):
    ax1=subplot(gs2[i],projection='polar')
    if i==1:
        ax1.set_title('Light_90deg Cue Rotation',fontsize=20)
    plot(tcurv_1[x])
    remove_polarAx(gca(),True)
    gca().set_yticks([])
    
for i,x in enumerate(cells1):
    ax1=subplot(gs2[i], projection='polar')
    b=plot(tcurv_2[x])
    remove_polarAx(gca(),True)
    gca().set_yticks([])
        

    
gs3=  GridSpecFromSubplotSpec(2,3, subplot_spec=gs[2])
for i,x in enumerate(spikes.keys()):
    ax2=subplot(gs3[i],projection='polar')
    if i==1:
        ax2.set_title('Light_90deg Floor Rotation',fontsize=20)
    plot(tcurv_1[x])
    remove_polarAx(gca(),True)
    gca().set_yticks([])
    
for i,x in enumerate(spikes.keys()):
    ax2=subplot(gs3[i], projection='polar')
    b=plot(tcurv_2[x])
    remove_polarAx(gca(),True)
    gca().set_yticks([])


     
gs4=  GridSpecFromSubplotSpec(2,3, subplot_spec=gs[3])     
 
    ############################################################################################### 
    # FIRING RATE ANALYSIS
    ###############################################################################################
    stats=findHDCells(tcurv_light,light_ep,spikes,position['ry'])
    hd_cells=stats['hd_cells']==True
    
    #mean_frate,peak_frate, pfd      = computeFiringRates(spikes,[light_ep, dark_ep],['light','dark'],[tcurv_light,tcurv_dark])    
    #info_hd                         = computeInfo([light_ep, dark_ep] ,spikes,position,['light','dark'])
    #mean_vLength                    = computeVectorLength(spikes,[light_ep,dark_ep], position['ry'],['light','dark'])
    spatial_corr                    = computeStability([light_ep,dark_ep],spikes,position['ry'],['light','dark'])
    #circ_mean,circ_var              = computeCircularStats([light_ep,dark_ep],spikes,position['ry'],['light','dark'])
    
    ###############################################################################################
    # CORRELATIONS (Auto & Cross)
    ###############################################################################################
    #autocorr_light, _             = compute_AutoCorrs(spikes, light_ep)
    #autocorr_dark, _              = compute_AutoCorrs(spikes, dark_ep)
    cc_light                      = compute_CrossCorrs(spikes, wake_ep_4)
    #cc_dark                       = compute_CrossCorrs(spikes, dark_ep)
    
    ################################################################################################
    # MERGE
    ################################################################################################
    all_means=all_means.append(mean_frate[hd_cells])                #Mean Firing Rates
    all_peaks=all_peaks.append(peak_frate[hd_cells])                #Peak Firing Rates
    all_pfd=all_pfd.append(pfd[hd_cells])                           #Preferred Firing Dir
    all_light_tcurv.append(tcurv_light[hd_cells])
    all_dark_tcurv.append(tcurv_dark[hd_cells])
    all_vLength=all_vLength.append(mean_vLength[hd_cells])          #Rayleigh Vector Length
    all_stability=all_stability.append(spatial_corr[hd_cells])      #Spatial Correlation Pearson(r)
    all_circMean=all_circMean.append(circ_mean[hd_cells])           #Circ Mean
    all_circVar=all_circVar.append(circ_var[hd_cells])              #Circ Variance
    all_info=all_info.append(info_hd[hd_cells])                     #Mutual Information

    ###############################################################################################
    # SAVING
    ###############################################################################################
    datatosave = {'sess_groups':sessions,'tcurves_light':all_light_tcurv,'tcurves_dark':all_dark_tcurv,
                  'pfd':all_pfd,'spatial_corr': all_stability,'mfrates':all_means,'info': all_info,
                  'pfrates':all_peaks,'vlength':all_vLength,'circMean':all_circMean, 'circVar':all_circVar}
    #data_files='C:/Users/kasum/Documents/HD_Drift/data'
    #datatosave.to_hdf(data_files+'/dark_data.h5',mode='a',key='dark_data') #save file


sys.exit()

cell1=pd.DataFrame(tmp[0])
cell1a=pd.DataFrame(tmp1[0])
cell1a.to_excel(r'C:\Users\kasum\Desktop\cell1a.xlsx')


corrcoef(ce)



A=np.diag(tmp[0],k=1)
B=np.diag(tmp1[0],k=1)


v=pd.DataFrame()
v1=pd.DataFrame()
for i in range(100):
    a= pd.DataFrame(np.diag(tmp[0],k=i))
    a1=pd.DataFrame(np.diag(tmp1[0],k=i))
    v=v.append(a)
    v1=v1.append(a1)
    
imshow(corrcoef(tmp[0],tmp[2]))


    if i==0:
        v=np.diag(tmp[0],k=i)
    else:
c=scipy.signal.correlate2d(tmp[0],tmp[0])  
    
    
    
    print(v)
    cA.append(v)




    