# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:45:05 2020

@author: kasum
"""

ep=ep2


    
pos=pd.DataFrame(index=(range(len(position.restrict(ep)))),columns=['x','z'])
pos['x']=position['x'].restrict(ep).values
pos['z']=position['z'].restrict(ep).values
x_cen=(pos['x'].max()+pos['x'].min())/2
y_cen=(pos['z'].max()+pos['z'].min())/2
cen=[x_cen,y_cen]

exp,dist=explore(ep,position)

r=np.sqrt((pos['x']-x_cen)**2+(pos['z']-y_cen)**2) #len of the radius at all points
cyl_r= r.max() #the radius of the area explored                   meters 56.2cm--cylinder size
cyl_c=cyl_r-cyl_r/2#0.10 2/3 of the cylinder 10cm from per

#####################################################################################
#Center Time
#####################################################################################
cen=r< cyl_c
pos_x_cen=position['x'].restrict(ep).index[cen]
pos_y_cen=position['z'].restrict(ep).index[cen]

starts=[]
ends=[]
start=pos_x_cen[0]

for i in range(len(pos_x_cen)-1):
    t=pos_x_cen[i]
    t1=pos_x_cen[i+1]
    d=t1-t
    if d>9000:
        starts.append(start)
        ends.append(t)
        start=t1
        
cen_eps=pd.DataFrame(data=[starts,ends], index=['start','end']).T 

## clean up to remove extremly small epochs
for i in range(len(cen_eps)):
    if diff(cen_eps.loc[i]) == 0:
        cen_eps=cen_eps.drop([i])
        
cen_eps=nts.IntervalSet(start=cen_eps['start'],end=cen_eps['end'])
tc_cen=computeAngularTuningCurves(spikes,position['ry'],cen_eps,60)

#############################################################################
#Wall Time
#############################################################################
wall=r>=cyl_c
pos_x_wall=position['x'].restrict(ep).index[wall]
pos_y_wall=position['z'].restrict(ep).index[wall]

starts=[]
ends=[]
start=pos_x_wall[0]

for i in range(len(pos_x_wall)-1):
    t=pos_x_wall[i]
    t1=pos_x_wall[i+1]
    d=t1-t
    if d>9000:
        starts.append(start)
        ends.append(t)
        start=t1
        
wall_eps=pd.DataFrame(data=[starts,ends], index=['start','end']).T 

## clean up to remove extremly small epochs
for i in range(len(wall_eps)):
    if diff(wall_eps.loc[i]) == 0:
        wall_eps=wall_eps.drop([i])
        
wall_eps=nts.IntervalSet(start=wall_eps['start'],end=wall_eps['end'])
tc_wall=computeAngularTuningCurves(spikes,position['ry'],wall_eps,60)


##############################################################################
##FIGURES
##############################################################################
#Path plots
figure()
for i in range(len(cen_eps)):
    ep_c=nts.IntervalSet(start=cen_eps.loc[i,'start'], end=cen_eps.loc[i,'end'])
    plot(position['x'].restrict(ep_c),position['z'].restrict(ep_c),c='r')
    
for i in range(len(wall_eps)):
    ep_w=nts.IntervalSet(start=wall_eps.loc[i,'start'], end=wall_eps.loc[i,'end'])
    plot(position['x'].restrict(ep_w),position['z'].restrict(ep_w),c='k')
    

#Tuning Curves
fig=figure()
fig.suptitle('Center Tuning Curves')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tc_cen[i],label=str(i),color='r', linewidth=2)
    ax2.set_xticklabels([])
    
    
fig=figure()
fig.suptitle('Wall Tuning Curves')
for i in spikes.keys():
    sz=(int(len(spikes.keys()))/4)+1
    ax2=subplot(sz,4,i+1, projection='polar')
    plot(tc_wall[i],label=str(i),color='k', linewidth=2)
    ax2.set_xticklabels([])
    
###############################################################################    
#STATISTICS
###############################################################################
#Tc width  
tcw_cen=tc_width(tc_cen,spikes) 
tcw_wall=tc_width(tc_wall, spikes)

figure()
boxplot([tcw_wall[0],tcw_cen[0]])
title('Tuning Curve Width')
scipy.stats.wilcoxon(tcw_cen[0],tcw_wall[0])


#info
tcInf_cen=hd_info(tc_cen, cen_eps,spikes,position)
tcInf_wall=hd_info(tc_wall, wall_eps,spikes,position)

figure()
boxplot([tcInf_wall.iloc[:,0],tcInf_cen.iloc[:,0]])
title('Information')

gca().set_ylim(0,2)

scipy.stats.wilcoxon(tcInf_wall.iloc[:,0],tcInf_cen.iloc[:,0])


#rates


