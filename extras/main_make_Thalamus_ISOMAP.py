#!/usr/bin/env python
'''
    File name: main_ripp_mod.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2


'''
import sys
import numpy as np
import pandas as pd
import scipy.io
from functions import *
# from pylab import *
# import ipyparallel
from multiprocessing import Pool
import os
import neuroseries as nts
from time import time
from pylab import *
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
from wrappers import *


data_directory = 'C:/Users/kasum/Desktop/KA30-190430'
#datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {ep:pd.DataFrame() for ep in ['wake', 'rem', 'sws']}

# session = 'Mouse17/Mouse17-130130'
session = '/KA30-190430'
# session = 'Mouse32/Mouse32-140822'

'''
shankStructure 	= loadShankStructure(generalinfo)
if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
else:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
'''

spikes,shank	= loadSpikeData(data_directory+session)		
n_channel,fs, shank_to_channel = loadXML(data_directory+session)	
wake_ep 		= loadEpoch(data_directory+session, 'wake')
#sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
#sws_ep 			= loadEpoch(data_directory+session, 'sws')
#rem_ep 			= loadEpoch(data_directory+session, 'rem')
#sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
#sws_ep 			= sleep_ep.intersect(sws_ep)	
#rem_ep 			= sleep_ep.intersect(rem_ep)
#rip_ep,rip_tsd 	= loadRipples(data_directory+session)
#rip_ep			= sws_ep.intersect(rip_ep)	
#rip_tsd 		= rip_tsd.restrict(sws_ep)
#speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)
#hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
#hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])


#spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
#neurons 		= np.sort(list(spikeshd.keys()))

# lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
# tmp = [lfp_hpc.loc[t-1e6:t+1e6] for i, t in enumerate(rip_tsd.index.values)]
# tmp = pd.concat(tmp, 0)
# tmp = tmp[~tmp.index.duplicated(keep='first')]		
# tmp.as_series().to_hdf(data_directory+session+'/'+session.split("/")[1]+'_EEG_SWR.h5', 'swr')

####################################################################################################################
# HEAD DIRECTION INFO
####################################################################################################################
'''#episodes = ['sleep','wake', 'sleep', 'wake','sleep']
#events = ['1','3']              #Index no of the wake episode

spikesth 		= {k:spikes[k] for k in np.where(shank < 5)[0]}
#neurons 		= np.sort(list(spikeshd.keys()))
position        = loadPosition(data_directory + session, events, episodes, n_ttl_channels = 1, optitrack_ch = 0)
'''
position = loadPosition(data_directory+session, 'wake')
angle 			= nts.Tsd(t = position.index.values, d = position['ry'].values, time_units = 'us')
#tcurves 		= computeAngularTuningCurves(spikeshd, angle, wake_ep.iloc[[0]], nb_bins = 61, frequency = 1/0.0256)

#wakangle        = angle.restrict(wake_ep.iloc[[0]])

####################################################################################################################
#ANGULAR TUNING CURVES CALCULATIONS AND HD DETECTION
####################################################################################################################
tuning_curves = computeAngularTuningCurves(spikes, angle, wake_ep.iloc[[3]], 61) ###restrict to specific wake ep

neurons 		= np.arange(len(tuning_curves.columns))
hd_index, stats = findHDCells(tuning_curves)

#hd_index = np.array([3,5,6,9])
#tuning_curves = tuning_curves[hd_index]
#tuning_curves = smoothAngularTuningCurves(tuning_curves, 10,2)
###################################################################################################################
#COLOR FILLING OF THE TUNING CURVES
###################################################################################################################
H = tuning_curves[0].index.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T

from matplotlib.colors import hsv_to_rgb
RGB = hsv_to_rgb(HSV)

for i, n in enumerate(tuning_curves.columns):
	color_l = hsv_to_rgb(np.array([(tuning_curves[n].idxmax()/(2*np.pi)),1, 1]))
	subplot(5,5,i+1, projection = 'polar')
	plot(tuning_curves[n], color = color_l)
	#scatter(tuning_curves[n].index.values, tuning_curves.values, c =RGB)	
show()

#sys.exit()
'''
fig, ax = plt.subplots(figsize = (6,1))
fig.subplots_adjust(bottom = 0.5)
cmap = matplotlib.cm.hsv
norm = matplotlib.colors.Normalize(vmin = 0, vmax = 360)
cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap = cmap, norm = norm, orientation = 'horizontal')
fig.show()
'''
####################################################################################################################
# binning data
####################################################################################################################

#rip_tsd = nts.Ts(rip_tsd.as_series().sample(500, replace = False).sort_index())
# rip_tsd = rip_tsd.iloc[0:200]

bins_size = [400,300,200,10,20]
allrates  = {}

####################################################################################################################
# BIN WAKE
####################################################################################################################

bin_size = bins_size[0]
bins = np.arange(wake_ep.as_units('ms').start.iloc[3], wake_ep.as_units('ms').end.iloc[3]+bin_size, bin_size)  # restrict the epoch with the corresp index
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

# allrates['wak'] = np.sqrt(spike_counts/(bins_size[0]*1e-3))
allrates['wak'] = np.sqrt(spike_counts/(bins_size[0]))

angle = angle.restrict(wake_ep.iloc[[3]]) #restrict the angle to corresp ep in wake
wakangle = pd.Series(index = np.arange(len(bins)-1))
tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
wakangle.loc[tmp.index] = tmp
wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)
#wakpos = pd.DataFrame(index = np.arange(len(bins)-1), columns = ['x','z'])
#tmp = postd.groupby(np.digitize(postd.as_units('ms').index.values, bins)-1).mean()
#wakpos.loc[tmp.index] = tmp
#wakpos.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

####################################################################################################################
# BIN SWS
####################################################################################################################
# bin_size = bins_size[3]
# tmp = []
# for start, end in zip(sws_ep.as_units('ms')['start'], sws_ep.as_units('ms')['end']):
# 	bins = np.arange(start, end+bin_size, bin_size)
# 	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
# 	for i in neurons:
# 		spks = spikeshd[i].as_units('ms').index.values
# 		spike_counts[i], _ = np.histogram(spks, bins)

# 	tmp.append(np.sqrt(spike_counts/(bins_size[1]*1e-3)))
# allrates['sws'] = tmp

####################################################################################################################
# BIN SWR
####################################################################################################################
'''
@jit(nopython=True)
def histo(spk, obins):
	n = len(obins)
	count = np.zeros(n)
	for i in range(n):
		count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
	return count


bin_size = bins_size[3]	
bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2

obins = np.vstack((bins-bin_size/2,bins)).T.flatten()
obins = np.vstack((obins,obins+bin_size)).T

# times = (np.arange(0, 2000+2*bins_size[-1], bins_size[-1]) - 1000 - bins_size[-1]/2)[0:-1] + bins_size[-1]/2
times = obins[:,0]+(np.diff(obins)/2).flatten()
tmp = []

rip_spikes = {}

for i, t in enumerate(rip_tsd.as_units('ms').index.values):
	print(i, t)
	tbins = t + obins
	# spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
	spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)
	rip_spikes[i] = {}
	for j in neurons:
		spks = spikeshd[j].as_units('ms').index.values
		# spike_counts[j], _ = np.histogram(spks, tbins)
		spike_counts[j] = histo(spks, tbins)
		nspks = spks - t
		rip_spikes[i][j] = nspks[np.logical_and((spks-t)>=-200, (spks-t)<=200)]
	# tmp.append(np.sqrt(spike_counts/(bins_size[0]*1e-3)))
	tmp.append(np.sqrt(spike_counts/(bins_size[-1])))
	# tmp.append(np.sqrt(spike_counts))
allrates['swr'] = tmp
'''

####################################################################################################################
# SMOOTHING
####################################################################################################################
tmp1 = allrates['wak'].rolling(window=400,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3).values

# tmp2 = []
# for rates in allrates['sws']:
# 	tmp2.append(rates.rolling(window=300,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=5).values)
# tmp2 = np.vstack(tmp2)


#tmp3 = []
#for rates in allrates['swr']:
#	tmp3.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=2).loc[-200:200].values)
#tmp3 = np.vstack(tmp3)

n = len(tmp1)

# n = 30000
# idx1 = np.random.choice(np.arange(len(tmp1)), n)
# idx2 = np.random.choice(np.arange(len(tmp2)), n)
# tmp = np.vstack((tmp1[idx1], tmp2[idx2], tmp3))
# tmp = np.vstack((tmp1[idx1], tmp3))
#tmp = np.vstack((tmp1, tmp3))



imap = Isomap(n_neighbors = 10, n_components = 2, n_jobs = -1).fit_transform(tmp1)


####################################################################################################################
# PLOTTING
####################################################################################################################
#iwak = imap[0:n]
# isws = imap[n:2*n]
# isws = imap[n:]
# iswr = imap[2*n:]
#isws = imap[n:]
#iswr = imap[n:]



#tokeep = np.where(np.logical_and(times>=-200,times<=200))[0]

#iswr = iswr.reshape(len(rip_tsd),len(tokeep),2)

#colors = np.hstack((np.linspace(0, 1, int(len(times)/2)), np.ones(1), np.linspace(0, 1, int(len(times)/2))[::-1]))[tokeep]

#colors = np.arange(len(times))[tokeep]

H = wakangle.values/(2*np.pi)

HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T

from matplotlib.colors import hsv_to_rgb
RGB = hsv_to_rgb(HSV)


figure()
scatter(imap[:,0], imap[:,1], c = RGB)
show()

sys.exit()			

def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.8          # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.set_xticks([])
	ax.set_yticks([])
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)


from matplotlib.gridspec import GridSpec
from matplotlib import colors


for i in range(iswr.shape[0]):
# for i in range(1):
	print(i)
	fig = figure(figsize = figsize(1.0))
	gs = GridSpec(5,1, figure = fig, height_ratios = [0.1, 0.0, 0.3, 0.2, 0.7], hspace = 0)

	ax0 = subplot(gs[0,:])
	noaxis(ax0)
	lfp = lfp_hpc.loc[rip_tsd.index[i]-3e5:rip_tsd.index[i]+3e5]
	lfp = nts.Tsd(t = lfp.index.values - rip_tsd.index[i], d = lfp.values)
	plot(lfp.as_units('ms'), color = 'black')
	plot([0], [lfp.max()-50], '*', color = 'red', markersize = 5)
	ylabel('CA1', labelpad = 25)
	xlim(times[tokeep][0], times[tokeep][-1])
	

	ax1 = subplot(gs[2,:])	
	simpleaxis(ax1)
	for j, n in enumerate(neurons):
		spk = rip_spikes[i][n]
		if len(spk):
			h = tcurves[n].idxmax()/(2*np.pi)
			hsv = np.repeat(np.atleast_2d(hsv_to_rgb([h,1,1])), len(spk), 0)
			scatter(spk, np.ones_like(spk)*j, c = hsv, marker = '|', s = 100, linewidth= 3)
	xlim(times[tokeep][0], times[tokeep][-1])
	ylim(-1, len(neurons)+1)
	xticks(np.arange(times[tokeep][0], times[tokeep][-1]+100, 100))
	xlabel("Time from SWR (ms)")
	ylabel("HD neurons")
	# ax1.spines['left'].set_visible(False)
	# plot([-300,-300], [0, len(neurons)], color = 'black')
	# x = times[tokeep[1:-1]]
	# y = np.ones_like(x)*-2
	# plot(x, y, '-', color = 'grey', zorder = 1)
	# scatter(x, y, c = np.arange(len(x)), zorder = 2, cmap=plt.cm.get_cmap('viridis'), s = 20)


	ax2 = subplot(gs[4,:])
	noaxis(ax2)
	ax2.set_aspect(aspect=1)
	scatter(iwak[~np.isnan(H),0], iwak[~np.isnan(H),1], c = RGB[~np.isnan(H)], marker = '.', alpha = 0.5, zorder = 2, linewidth = 0, s= 40)
	plot(iswr[i,:,0], iswr[i,:,1], alpha = 0.5, zorder = 4, color = 'grey')
	# cNorm = colors.Normalize(vmin = 0, vmax=1)
	cl = np.linspace(0, 0.7, len(tokeep))
	scatter(iswr[i,:,0], iswr[i,:,1], c =  cl, zorder = 5, cmap=plt.cm.get_cmap('bone'), s = 50, vmin = 0, vmax = 1.0)
	idx = np.where(times[tokeep] == 0)[0][0]
	plot(iswr[i,idx,0], iswr[i,idx,1], '*', color = 'red', zorder = 6, markersize = 10)
	# ax2.set_title("Head-direction manifold")
	ylabel(session+ " " + str(rip_tsd.as_units('s').index.values[i]) + " s " + str(i), fontsize = 6, labelpad = 40)

	# hsv
	display_axes = fig.add_axes([0.2,0.45,0.1,0.1], projection='polar')
	colormap = plt.get_cmap('hsv')
	norm = mpl.colors.Normalize(0.0, 2*np.pi)
	xval = np.arange(0, 2*pi, 0.01)
	yval = np.ones_like(xval)
	display_axes.scatter(xval, yval, c=xval, s=100, cmap=colormap, norm=norm, linewidths=0, alpha = 0.8)
	display_axes.set_yticks([])
	display_axes.set_xticks(np.arange(0, 2*np.pi, np.pi/2))
	display_axes.grid(False)

	#colorbar	
	c_map_ax = fig.add_axes([0.44, 0.59, 0.22, 0.02])
	# c_map_ax.axes.get_xaxis().set_visible(False)
	# c_map_ax.axes.get_yaxis().set_visible(False)
	# c_map_ax.set_xticklabels([times[tokeep][0], 0, times[tokeep][-1]])
	cb = mpl.colorbar.ColorbarBase(c_map_ax, cmap=plt.cm.get_cmap('bone'), orientation = 'horizontal')
	cb.ax.set_xticklabels([int(times[tokeep][0]), 0, int(times[tokeep][-1])])
	# cb.ax.set_title("Time from SWR (ms)", fontsize = 7)

	gs.update(left = 0.15, right = 0.95, bottom = 0.05, top = 0.95)

	savefig("../figures/figures_articles_v4/figure1/ex_swr_"+str(i)+".pdf", dpi = 900, facecolor = 'white')
	# os.system("evince ../figures/figures_articles_v4/figure1/ex_swr_9.pdf &")

os.system("pdftk ../figures/figures_articles_v4/figure1/ex_swr_*.pdf cat output ../figures/figures_articles_v4/figure1/swr_all_exemples.pdf")
os.system("evince ../figures/figures_articles_v4/figure1/swr_all_exemples.pdf &")
os.system("rm ../figures/figures_articles_v4/figure1/ex_swr_*")



guimes version for multiple manifolds on one fig

def makeRingManifold(spikes, ep, angle, bin_size = 200):
    """
    spikes : dict of hd spikes
    ep : epoch to restrict
    angle : tsd of angular direction
    bin_size : in ms
    """
    neurons = np.sort(list(spikes.keys()))
    inputs = []
    angles = []
    sizes = []
    for j in ep.index.values:
        bins = np.arange(ep.as_units('ms').start.iloc[j], ep.as_units('ms').end.iloc[j]+bin_size, bin_size)
        spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
        
        for i in neurons:
            spks = spikes[i].as_units('ms').index.values
            spike_counts[i], _ = np.histogram(spks, bins)
    
        rates = np.sqrt(spike_counts/(bin_size))
        
        epi = nts.IntervalSet(ep.loc[j,'start'], ep.loc[j,'end'])
        angle2 = angle.restrict(epi)
        newangle = pd.Series(index = np.arange(len(bins)-1))
        tmp = angle2.groupby(np.digitize(angle2.as_units('ms').index.values, bins)-1).mean()
        tmp = tmp.loc[np.arange(len(bins)-1)]
        newangle.loc[tmp.index] = tmp
        newangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)
    
        tmp = rates.rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2).values
        sizes.append(len(tmp))
        inputs.append(tmp)
        angles.append(newangle)

    inputs = np.vstack(inputs)
    

    imap = Isomap(n_neighbors = 20, n_components = 2, n_jobs = -1).fit_transform(inputs)    
    
    # if more than 2, come back here and add sizes[0]:sizes[1]
    imaps = [imap[0:sizes[0]], imap[sizes[0]:]]
    
    RGBs  = []
    Hs = []
    for i, a in enumerate(angles):
        H = a.values/(2*np.pi)
        HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
        RGB = hsv_to_rgb(HSV)
        RGBs.append(RGB)
        
    figure()
    for i in range(len(imaps)):
        ax = subplot(1,len(imaps),i+1)
        ax.set_aspect(aspect=1)
        iwak = imaps[i]
        RGB = RGBs[i]
        ax.scatter(iwak[:,0], iwak[:,1], c = RGB, marker = 'o', alpha = 0.5, zorder = 2, linewidth = 0, s= 40)    
    
    # hsv
    '''display_axes = fig.add_axes([0.2,0.45,0.1,0.1], projection='polar')
    colormap = plt.get_cmap('hsv')
    norm = mpl.colors.Normalize(0.0, 2*np.pi)
    xval = np.arange(0, 2*pi, 0.01)
    yval = np.ones_like(xval)
    display_axes.scatter(xval, yval, c=xval, s=100, cmap=colormap, norm=norm, linewidths=0, alpha = 0.8)
    display_axes.set_yticks([])
    display_axes.set_xticks(np.arange(0, 2*np.pi, np.pi/2))
    display_axes.grid(False)'''
    
    show()
    
    return imaps 