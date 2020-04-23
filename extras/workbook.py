# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:48:53 2020

@author: kasum
"""
light=PFD_Rates(ep2,spikes,position['ry'], 1e+6)
    dark=PFD_Rates(ep1,spikes,position['ry'], 1e+6)        
    
    drift=pd.DataFrame(columns=[cond2,cond1])

    cond=dark
    for i in spikes:
        idx=cond[1][i].values >= tcurv_1[i].max() * 0.70
        x=array(cond[0].index[idx]).reshape(-1,1)
        y=array(np.unwrap(cond[0][i][idx].values)).reshape(-1,1)
        regr=linear_model.LinearRegression()
        regr.fit(x,y)
        reg_pred=regr.predict(y)
        drift.loc[i,cond1]=regr.coef_
        
    cond=light
    for i in spikes.keys():
        idx=cond[1][i].values >= tcurv_2[i].max() * 0.70
        x=array(cond[0].index[idx]).reshape(-1,1)
        y=array(np.unwrap(cond[0][i][idx].values)).reshape(-1,1)
        regr=linear_model.LinearRegression()
        regr.fit(x,y)
        reg_pred=regr.predict(y)
        drift.loc[i,cond2]=regr.coef_
    
    all_drift=all_drift.append(drift[hd])
              


data_files='C:/Users/kasum/Documents/HD_Drift/data'
ka41_drift_files.to_hdf(data_files+'/ka41_drift_files.h5',key='ka41_drift_files' )     
        
    ka41_light=light
ka41_dark=dark    
    
ka41_drift_files=pd.DataFrame({'merged': drift_dataset,'light':light,'dark':dark} )
    
    
    
combined_drift=all_drift
drift_dataset.to_hdf(data_files+'/drift_dataset.h5',mode='a',key='drift_dataset')    
    
    
drift_dataset=pd.DataFrame(index=np.arange(len(all_drift)), columns=['light','dark'])    
for i in range(len(all_drift)):
    drift_dataset.iloc[i, 0]=float(abs(all_drift.iloc[i,0][0]))
    drift_dataset.iloc[i, 1]=float (abs(all_drift.iloc[i,1][0]))
        
all_drift
    
    
    
    
    
    
    
    
    subplot(projection='polar');plot(tcurv_1[3])
    
    
    
def computeAngularTuningCurves(spikes, angle, ep, nb_bins = 180, frequency = 120.0):

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
        tuning_curves[k] = spike_count





angle=position['ry']
ep=wake_ep_2

dur=1e+6


def PFD_sw(ep,spikes,position,dur): #duration must be in microsecs
    sw_ep=slidingWinEp(ep,dur)       
    max_tcurves=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    max_pRate=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    for i in range(len(sw_ep)):
        sw=sw_ep.loc[i]
        sw=nts.IntervalSet(sw.start,sw.end)
        tcurve=computeAngularTuningCurves(spikes,position,sw,60)
        for k in spikes.keys():
            pFD=tcurve[k].idxmax(axis=0)
            pFR=tcurve[k].max()
            max_pRate.iloc[i,k]=pFR
            max_tcurves.iloc[i,k]=pFD
    return max_tcurves, max_pRate










































ep=ep1
figure()
for i,x in enumerate(range(15)):
    ax=subplot(gs[i])
    ang_spk=position['ry'].realign(spikes[i].restrict(ep)).as_units('s')
    #ang=np.unwrap(position['ry'].realign(spikes[i].restrict(ep)).as_units('s').values)
    ang=ang_spk.values
    #ang=np.unwrap(ang*(180/np.pi)) #degrees
    spk=ang_spk.index
    m,c=np.polyfit(array(spk),array(ang),1)
    
    scatter(spk,ang, s=8,c='k', alpha=0.5)
    
    #plot(np.unwrap(position['ry'].restrict(ep).as_units('s')))
    gca().set_ylabel('Head Direction (rad)',size=9)
    gca().set_xlabel('Time (s)',size=9)
    gca().tick_params(labelsize=10)
    gca().set_ylim(-50,40)
    #gca().set_ylim(-80,80)#used for unwrapped angle
    gca().spines['right'].set_visible(False)
    gca().spines['top'].set_visible(False)

    linearline=plt.plot(array(spk), m*array(spk)+c, '--k', linewidth=2, zorder=1)
    
    
rad2deg((pf[1]-pf[0]) % np.pi)
abs((pf[0] *(180/np.pi))-(pf[1] *(180/np.pi)))


rad2deg(arctan2(sin(6.01 - 0.57), cos(6.01 - 0.57)))




np.unwrap(array(pf[0],6.12,0.1))

   
   np.arctan2(sin(2.39613),cos(6.128))*(180/np.pi)


pf[0] *(180/np.pi)


#############################
#Extra Drift Analysis
###########################

d_t=array(ang_spk.index[1:]-ang_spk.index[:-1])

d_a=arctan2(sin(ang_spk.values[1:] - ang_spk.values[:-1]), cos(ang_spk.values[1:] - ang_spk.values[:-1]))


d_a1=(ang_spk.values[1:] - ang_spk.values[:-1])%np.pi

n=pd.DataFrame(index=(np.arange(len(d_t))),columns=['log_t','log_a'])


n['log_t']=d_t
n['log_a']=d_a
#n=pd.DataFrame(log(d_t),log(d_a))

n1=log(n)


avg_ang=mean(d_a)
dur=ep.tot_length('s')
drift=print(avg_ang/dur)

