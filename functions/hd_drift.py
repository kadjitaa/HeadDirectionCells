# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:20:20 2019

@author: kasum
"""
import pandas as pd
from matplotlib.gridspec import gridspec
gs=GridSpec(3,4)

###############################################################################
## DRIFT WRAPPED ANGLE  #####GIVES a more intuitive view of the drift but not reliable 
#estimator of the direction of drift
###############################################################################
ep=wake_ep_2
figure()

fig,ax=subplots(figsize=(6.48,5.59))
for i,x in enumerate(range(9)):
    ax=subplot(gs[i])
    ang_spk=position['ry'].realign(spikes[i].restrict(ep)).as_units('s')
    ang=ang_spk.values
    spk=ang_spk.index
    m,c=np.polyfit(array(spk),array(ang),1)
    scatter(spk,ang,zorder=3, s=2,c='r', alpha=0.5)
    
    plot(position['ry'].restrict(ep).as_units('s'), color='grey', alpha=0.5,linewidth=1)
    gca().set_ylim(0,2*np.pi)
    
    #gca().set_ylabel('Head Direction (rad)',size=9)
    #gca().set_xlabel('Time (s)',size=9)
    gca().tick_params(labelsize=10)
    gca().spines['right'].set_visible(False)
    gca().spines['top'].set_visible(False)
    linearline=plt.plot(array(spk), m*array(spk)+c, '--k', linewidth=2, zorder=1)

###############################################################################
## DRIFT UNWRAPED ANGLE
###############################################################################
ep=wake_ep_1
figure()
subplot(211)
#gs=GridSpec(3,4)
for i,x in enumerate(range(9)):
   # ax=subplot(gs[i])
    ang2=unwrap(position['ry'].restrict(ep).values)
    ang2=nts.Tsd(t=position['ry'].restrict(ep).index.values, d=ang2) 
    ang_spk=ang2.realign(spikes[i].restrict(ep)).as_units('s')
    ang=ang_spk.values    
    spk=ang_spk.index.values
    scatter(spk,ang,s=2)
    gca().set_xticks([])
    gca().set_ylabel('Unwrapped Head Direction (rad))

subplot(212)
plot(ang2,color='grey')
    




#############################################################################
####DRIFT SOLUTION 2 ###works
#############################################################################
#ep=wake_ep_2
#dur=1e+6    #bin spikes into 1sec windows
#position=position['ry']

def PFD_Rates(ep,spikes,position,dur): #duration must be in microsecs
    sw_ep=slidingWinEp(ep,dur)       
    max_tcurves=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    max_pRate=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    for i in range(len(sw_ep)):
        sw=sw_ep.loc[i]
        sw=nts.IntervalSet(sw.start,sw.end)
        tcurve=computeFrateAng(spikes,position,sw,60)
        for k in spikes.keys():
            pFD=tcurve[k].idxmax(axis=0)
            pFR=tcurve[k].max()
            max_pRate.iloc[i,k]=pFR
            max_tcurves.iloc[i,k]=pFD
    return max_tcurves, max_pRate
#################################################################################
#Plots
#################################################################################

light=PFD_Rates(wake_ep_2,spikes,position['ry'], 1e+6)
dark=PFD_Rates(wake_ep_1,spikes,position['ry'], 1e+6)

cond=dark


gs=GridSpec(3,4)
coefs=[]
figure()
for i in spikes.keys():
    ax=subplot(gs[i])
    idx=cond[1][i].values >= tuning_curves_1[i].max() * 0.70
    
    #hist(cond[1][i].values[idx])
    
    scatter(cond[0][i].index.values[idx], np.unwrap(cond[0][i][idx].values), label=str(i))
    #Regression Fit Line
    sns.set_style(style='white')
    sns.regplot( x=array(cond[0].index[idx]),y=array(np.unwrap(cond[0][i][idx].values)), line_kws={'color':'red'}) #scatter_kws can also be used to change the scatter color
    legend(['linear_fit','spikes'])
    #gca().set_ylabel('Unwrapped Head Direction (rad)')
    gca().set_xlabel('Time (s)')
    gca().spines['right'].set_visible(False)
    gca().spines['top'].set_visible(False)
    gca().set_ylim(0,2*np.pi)
    
    #Regression Stats
    x=array(cond[0].index[idx]).reshape(-1,1)
    y=array(np.unwrap(cond[0][i][idx].values)).reshape(-1,1)
    regr=linear_model.LinearRegression()
    
    regr.fit(x,y)
    reg_pred=regr.predict(y)
    coefs.append(regr.coef_)
    
    #Pearson correlation
    scipy.stats.pearsonr(x[:,0],array(y[:,0]))



for i in spikes.keys():
    l_idx=light[1][i].values >= tuning_curves_2[i].max() * 0.50
    a=np.unwrap(light[0][i][l_idx].values)
    plot(np.cumsum(np.diff(a)),color='r')
        
    d_idx=dark[1][i].values >= tuning_curves_1[i].max() * 0.50
    a1=np.unwrap(dark[0][i][d_idx].values)
    plot(np.cumsum(np.diff(a1)),color='k')

cumulative change in dark vs light; interpolate and fit a line to the change

slope/duration of the epoch.... speed in rad/s





###############################################################


figure()
for i in spikes.keys():
    ax=subplot(gs[i])
    idx=dark_pRate[i].values >= tuning_curves_1[i].max() * 0.75 #creates an index of all locations with frate >= 75% of the peak firing rate of the cell 
    scatter(dark_tcurve.index[idx], np.unwrap(dark_tcurve[i][idx].values),label=str(i)) 
    
    #Regression Fit Line
    sns.set_style(style='white')
    sns.regplot( x=array(dark_tcurve.index[idx]),y=array(np.unwrap(dark_tcurve[i][idx].values)), line_kws={'color':'red'}) #scatter_kws can also be used to change the scatter color
    legend(['linear_fit','spikes'])
    gca().set_ylabel('Unwrapped Head Direction (rad)')
    gca().set_xlabel('Time (s)')
    gca().spines['right'].set_visible(False)
    gca().spines['top'].set_visible(False)
    
    
    #Regression Stats
    x=array(dark_tcurve.index[idx]).reshape(-1,1)
    y=array(np.unwrap(dark_tcurve[i][idx].values)).reshape(-1,1)
    regr=linear_model.LinearRegression()
    regr.fit(x,y)
    reg_pred=regr.predict(y)
    regr.coef_
 
    #Pearson correlation
    scipy.stats.pearsonr(x[:,0],array(y[:,0]))



a=np.unwrap(dark_tcurve[i][idx].values)
a1=np.cumsum(abs(np.diff(a)))














from sklearn.linear_model import LinearRegression
figure();
for i in spikes.keys():
    ax=subplot(gs[i])
    idx=max_pRate[i].values >= tuning_curves_2[i].max() * 0.50 #creates an index of all locations with frate >= 75% of the peak firing rate of the cell 
    
  
    scatter(max_tcurves.index[idx], np.unwrap(max_tcurves[i][idx].values), label=str(i))
    sns.regplot( x=array(max_tcurves.index[idx]),y=array(np.unwrap(max_tcurves[i][idx].values)), line_kws={'color':'red'}) #scatter_kws can also be used to change the scatter color

    gca().set_ylim(0,15)   
    x=array(max_tcurves.index[idx])#.reshape(-1,1)
    y=array(np.unwrap(max_tcurves[i][idx].values))#.reshape(-1,1))
    scipy.stats.pearsonr(x,y)
    reg= LinearRegression().fit(x,y)    
    reg.score(x, y)
    reg.coef_
    
    
    
    
    m,c=np.polyfit(array(max_tcurves.index[idx][:50]),array(max_tcurves[i][idx][:50].astype('float')),1)
    
    linearline=plt.plot(array(max_tcurves.index[idx][:50]), m*(max_tcurves[i][idx][:50])+c, 'k', linewidth=2, zorder=1)
    legend()
    gca().set_ylim(0,2*np.pi)


dark_pRate=max_pRate
dark_tcurve=max_tcurves


from sklearn import linear_model
















###############################################################################
##DRIFT ESTIMATION WITH BAYSIAN DECODING
###############################################################################
#decoded angle
#Define Training Data & Epoch
Epoch= wake_ep_2#wake_ep_2_ka30 #epoch to be decoded
train_ep=nts.IntervalSet(start=Epoch.loc[0,'start'], end =Epoch.loc[0,'start']+1.2e+8) #2mins provides the best decoder of final PFD across all cells
tc_train=computeAngularTuningCurves(spikes,position ['ry'],train_ep,60) # tcurves for training decoder

################################################################################
'''
figure();
for i in spikes.keys():
    subplot(3,4,i+1,projection='polar'); plot(tc_train[i], label=str(i)); legend()
'''
######################################################
#run baysian decoder
######################################################
decoded_pos,ang=decodeHD(tc_train,spikes, Epoch) 
#decoded_pos=pd.DataFrame(decoded_pos)
actual_pos=position['ry'].restrict(Epoch)


def makeBins(ep, bin_size=200): #the bin size is based on the bin size of the decoder
    bins_=  np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)      
    return bins_

bins = makeBins(Epoch)
index = np.digitize(actual_pos.as_units('ms').index.values, bins)-1
down_actual_pos = actual_pos.groupby(index).mean()  # here you are taking the mean of the positions corresponding to each unique binned index
down_actual_pos = nts.Tsd(t = bins[0:-1]+np.diff(bins)/2, d =down_actual_pos.values[0:len(bins)-1], time_units = 'ms')


true_ang=np.cumsum(np.abs(np.diff(np.unwrap(down_actual_pos.values))))
dec_ang=np.cumsum(np.abs(np.diff(np.unwrap(decoded_pos.values))))



figure();
plot(true_ang,color='k',linestyle='-'); plot(dec_ang, color='r',linestyle='-')
gca().set_ylim(0,1500)


figure();
plot(decoded_pos); plot(down_actual_pos, color='r')


#Compute Decoding Error
decoded_err=np.arctan2(np.sin(down_actual_pos.values-decoded_pos.values),np.cos(down_actual_pos.values-decoded_pos.values))
#mean_decoded_err=print(np.abs(decoded_err).mean())


figure();plot(cumsum(abs(decoded_err)),color='r')



#########################################
#DRIFT ANALYSIS
########################################

eps= full_ang(wake_ep_1,position['ry'])

t=pd.DataFrame(index=np.arange(len(eps)), columns=['t']) #time is in microseconds
for i in range(len(eps)):
    t.loc[i]=eps.iloc[i,0]+(diff(eps.loc[i])/2)  #this corresponds to delta t
      
    
#ang=pd.DataFrame(index=np.arange(len(eps)),columns=['ang'])  
pfd=pd.DataFrame(index=t['t'].values,columns=spikes.keys())    
for i in range(len(eps)):    
    ep=nts.IntervalSet(start=eps.loc[i,'start'], end =eps.loc[i,'end'])
    tc=computeAngularTuningCurves(spikes,position ['ry'],ep,60)
    for spk in spikes.keys():
         pfd.iloc[i,spk]=tc[spk].idxmax()


drift= pd.DataFrame(index=[0],columns=spikes.keys())
dur=array(pfd.index)
figure()
for i in drift.columns:
    c_ang=pfd[i].values.astype('float')
    d_ang=abs(arctan2(sin(c_ang[1:]-c_ang[:-1]),cos(c_ang[1:]-c_ang[:-1])))
    drift.iloc[0,i]=average(c_ang/dur)*1e+6 #converts the drift to seconds
    subplot(3,3,i+1)
    scatter(np.arange(len(d_ang)),d_ang)
    plot(d_ang)
    gca().set_ylim(0,2*np.pi)

'''
ToDo
unwrap the angles or use arctan to find the diff
'''

av_ang_drift=mean(d_ang) 
av_t_drift=mean(d_time.values[1:])

figure()
for i in spikes.keys():
    subplot(4,3,1+i)
    scatter(t.t,pfd[i])










def computeFrateAng(spikes, angle, ep, nb_bins = 180, frequency = 120.0):
    '''Computes the ang tcurves without normalising to occupancy.
    It will essentiall give you the total spike count for each angular position
    '''

    bins             = np.linspace(0, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    tuning_curves     = pd.DataFrame(index = idx, columns = np.arange(len(spikes)))    
    angle             = angle.restrict(ep)
    # Smoothing the angle here
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
    tmp2             = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
    angle            = nts.Tsd(tmp2%(2*np.pi))
    for k in spikes:
        spks             = spikes[k]
        # true_ep         = nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), end = np.minimum(angle.index[-1], spks.index[-1]))        
        spks             = spks.restrict(ep)    
        angle_spike     = angle.restrict(ep).realign(spks)
        spike_count, bin_edges = np.histogram(angle_spike, bins)
        occupancy, _     = np.histogram(angle, bins)
        tuning_curves[k] = spike_count    

    return tuning_curves























'''NOTES
1.Disambiguate the drift of the cells from the head direction
2.I can use the sliding win approach to estimate the time it takes for a cells to stabilize
3. you can get the final PFD after about 77sec of recording