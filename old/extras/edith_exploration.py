# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:29:32 2019

@author: kasum
"""
import numpy as np
import pandas as pd
from pylab import *
#solution
pos=pd.read_excel(r'C:\Users\kasum\Downloads\M2-c2.xlsx',nrows=500)
dx=pos.diff()
val=pos.X[0]
for i in range(len(pos)):
    if abs(val-abs(pos.X[i]))<30:
        val= pos.X[i]
    else:
        pos.X[i]=pos.X[i-1]
        #pos.Y[i]=pos.Y[i-1]

vals=pos.Y[0]
for i in range(len(pos)):
    if abs(vals-abs(pos.Y[i]))<30:
        vals= pos.Y[i]
    else:
        #pos.X[i]=pos.X[i-1]
        pos.Y[i]=pos.Y[i-1]        
     
 ######################################################       
pos_=pd.read_excel(r'C:\Users\kasum\Downloads\M2-c2.xlsx')


pos_x=array(pos_['X'])
pos_y=array(pos_['Y'])
        
dx = pos_x[1:]-pos_x[:-1]
dy = pos_y[1:]-pos_y[:-1]
dist = np.concatenate(([0],np.sqrt(dx**2+dy**2)))

for i,x in enumerate (dist):
    if x >100: 
        pos_.iloc[i,0]=NaN #pos_.iloc[i-1,0]
        pos_.iloc[i,1]=NaN #pos_.iloc[i-1,1]

               
        
        

figure();plot(pos_['X'],pos_['Y'])



def secondl(a):
    largest=min(a)
    sclargest=[]
    for i in a:
        if i>largest:
            sclargest=largest
            largest=i
    return sclargest



       
#Define position as an array
pos_x=array(pos_['X'])
pos_y=array(pos_['Y'])

fr= 120 #camera frame rate

#you may have to manually define the center of your environment for data with irregular path plots
x_cen=(pos_x.max()+pos_x.min())/2  
y_cen=(pos_y.max()+pos_y.min())/2
#c_vert=plot([x_cen,x_cen], [pos_y.min(),pos_y.max()]) #plots vertical line through center for verification
#c_hor=plot([pos_x.min(),pos_x.max()],[y_cen,y_cen])   #plots horizontal line through center for verification

#DISTANCE TRAVELLED
dx = pos_x[1:]-pos_x[:-1]
dy = pos_y[1:]-pos_y[:-1]
dist = np.concatenate(([0],np.sqrt(dx**2+dy**2)))  #computes the distance between two consecuitive x,y points

#ZONES
#upper left
up_left=(x_cen>pos_x) & (y_cen<pos_y) #defining quadrant
up_left_allDis=dist[up_left] #Extracting distance covered in quadrant
up_left_totDis=sum(up_left_allDis) #distance covered
up_left_totTime=len(up_left_allDis)/fr  #total time in seconds
up_left_vel=up_left_totDis/up_left_totTime  #velocity in quadrant

#upper right
up_right=(x_cen<pos_x) & (y_cen<pos_y)
up_right_allDis=dist[up_right] #Extracting distance covered in quadrant
up_right_totDis=sum(up_right_allDis) #distance covered
up_right_totTime=len(up_right_allDis)/fr  #total time in seconds
up_right_vel=up_right_totDis/up_right_totTime  #velocity in quadrant

#buttom left
b_left=(x_cen>pos_x) & (y_cen>pos_y)
b_left_allDis=dist[b_left] #Extracting distance covered in quadrant
b_left_totDis=sum(b_left_allDis) #distance covered
b_left_totTime=len(b_left_allDis)/fr  #total time in seconds
b_left_vel=b_left_totDis/b_left_totTime  #velocity in quadrant

#lower right
b_right=(x_cen<pos_x) & (y_cen>pos_y)
b_right_allDis=dist[b_right] #Extracting distance covered in quadrant
b_right_totDis=sum(b_right_allDis) #distance covered
b_right_totTime=len(b_right_allDis)/fr  #total time in seconds
b_right_vel=b_right_totDis/b_right_totTime  #velocity in quadrant

tot_dist=sum([up_left_d,up_right_d,b_left_d,b_right_d])

###############################################################################
#PLOTS
###############################################################################
figure(); plot(pos_x,pos_y)  #Full covereage

quad=b_right #define quadrant e.g up_right, up_left, b_left

p_x=pos_x[quad]
p_y=pos_y[quad] 
plot(p_x,p_y) # plot defined quadrant





##############################################################################
#COMPRESSED CODE
##############################################################################
from astropy.visualization import hist
pos_=pd.read_excel(r'C:\Users\kasum\Downloads\M2-c2.xlsx')
pos_x=array(pos_['X'])
pos_y=array(pos_['Y'])


#DISTANCE TRAVELLED
dx = np.concatenate(([0],pos_x[1:]-pos_x[:-1]))
dy = np.concatenate(([0],pos_y[1:]-pos_y[:-1]))

dist = np.sqrt(dx**2+dy**2)  #computes the distance between two consecuitive x,y points
hist(dist,bins='knuth')


new_pos=pd.DataFrame(index=np.arange(len(pos_)),columns=['x','y'])

#1st Cleaning using x


pos=pd.read_excel(r'C:\Users\kasum\Downloads\M2-c2.xlsx', nrows=13)



pos_=position.iloc[:20,[0,1]]
pp_=pd.DataFrame(index=np.arange(len(pos_)),columns=['X','Y'])
for times in range(5):    
    pos_x=array(pos_['X'])
    #dx = abs(np.concatenate(([0],pos_x[1:]-pos_x[:-1])))  
    dx=diff(pos_['X'])    
    for i in range(len(dx)):
        if dx[i] > 30:
            pp_.iloc[i,0]=pos_.iloc[i-1,0]

        
        
p=np.concatenate(([0],diff(pos_['X'])))    
for i in range(len(p)):
    if abs(p[i])>30:
        pos_.iloc[i,0]=pos_.iloc[i-1,0]
        
                
        
        
        
        
        
        else:
            pos_.iloc[i,0]=pos_.iloc[i,0]
            pos_.iloc[i,1]=pos_.iloc[i,1]

subplot(211)
plot(pos['X'], pos['Y'])

subplot(211)        
plot(pos_['X'], pos_['Y'],color='r')


        
        
        
        
        if abs(dy[i]) > 6:
            new_pos.iloc[i,0]=pos_.iloc[i-1,0]
            new_pos.iloc[i,1]=pos_.iloc[i-1,1]
        else:
           new_pos.iloc[i,0]=pos_.iloc[i,0]
           new_pos.iloc[i,1]=pos_.iloc[i,1]


    
    
    
    
    if dy[i]> 0.5:
        new_pos.iloc[i,0]=pos_.iloc[i-1,0]
        new_pos.iloc[i,1]=pos_.iloc[i-1,1]


figure(); plot(new_pos['x'],new_pos['y'])



new_pos=pd.DataFrame(index=np.arange(len(pos_)),columns=['x','y'])

for i in range(len(pos_)):
    if dist[i]>17:
        new_pos.iloc[i,0]=pos_.iloc[i-1,0]
        new_pos.iloc[i,1]=pos_.iloc[i-1,1]
        
        
        
        
    else:
        new_pos.iloc[i,0]=pos_.iloc[i,0]
        new_pos.iloc[i,1]=pos_.iloc[i,1]

figure(); plot(new_pos['x'],new_pos['y'])


x=np.digitize(dist,bins=np.arange(186))



np.linspace()











#DEFINE ZONES
up_left=(x_cen>pos_x) & (y_cen<pos_y)
up_right=(x_cen<pos_x) & (y_cen<pos_y)
b_left=(x_cen>pos_x) & (y_cen>pos_y)
b_right=(x_cen<pos_x) & (y_cen>pos_y)

quads=[up_left,up_right, b_left, b_right]

data=pd.DataFrame(index=(['up_left', 'up_right', 'b_left', 'b_right']), columns=(['tot_dist','tot_time','vel']))
for i,x in enumerate (range(len(quads))):
    allDis=dist[quads[i]] #Extracting distance covered in quadrant
    totDis=sum(allDis) #distance covered   MOdify this to reflect the appropriate units
    totTime=len(allDis)/fr  #total time in seconds
    vel=totDis/totTime  #velocity in quadrant
    data.iloc[i,2]=vel
    data.iloc[i,0]=totDis
    data.iloc[i,1]=totTime
    
    
    
    
    






































from astropy.visualization import hist
pos_=pd.read_excel(r'C:\Users\kasum\Downloads\M2-c2.xlsx')
pos_x=array(pos_['X'])
pos_y=array(pos_['Y'])


#DISTANCE TRAVELLED
dx = np.concatenate(([0],pos_x[1:]-pos_x[:-1]))
dy = np.concatenate(([0],pos_y[1:]-pos_y[:-1]))
#dist = np.sqrt(dx**2+dy**2)  #computes the distance between two consecuitive x,y points

new_pos=pd.DataFrame(index=np.arange(len(pos_)),columns=['x','y'])
##################################################################


for i in range(len(pos_)):
    dx[i]=abs(pos_x[1:]-pos_x[:-1])
    if dx[i] > 6:
        pos_x[i]
        
        
     
pos_=pd.read_excel(r'C:\Users\kasum\Downloads\M2-c2.xlsx')
pos_x=array(pos_['X'])
pos_y=array(pos_['Y'])
dx = np.concatenate(([0],pos_x[1:]-pos_x[:-1]))
dy = np.concatenate(([0],pos_y[1:]-pos_y[:-1]))       

while dx[i] > 6:


def dis(pos_):
    pos_x=array(pos_['X'])
    dx=np.concatenate(([0],pos_x[1:]-pos_x[:-1]))
    return dx

def replace(pos_):
    for i in pos_:
        if dis(pos_)[i]>6:
            pos.iloc[i,0]=pos_.iloc[i-1,0]
        else:
            pos.iloc[i,0]=pos_.iloc[i,0]
            


   


dx=dis(pos_)

while dx>6
pos=pd.DataFrame(index=np.arange(len(pos_)),columns=['X','Y'])    
for i,x in enumerate (bo):
    if x==True:
        pos.iloc[i,0]=pos_.iloc[i-1,0]
    elif x==False:
        pos.iloc[i,0]=pos_.iloc[i,0]    
a=dis(pos)        




    
    
def computeDist (pos_):
    pos=pd.DataFrame(index=np.arange(len(pos_)),columns=['X','Y'])
    pos_x=array(pos_['X'])
    pos_y=array(pos_['Y'])
    #dy = np.concatenate(([0],pos_y[1:]-pos_y[:-1]))
    for i in range(len(pos_)):
        dx = np.concatenate(([0],pos_x[1:]-pos_x[:-1]))
        while any (dx>6):
            pos.iloc[i,0]=pos_.iloc[i-1,0]
            posx=array(pos_['X'])
            dx=np.concatenate(([0],posx[1:]-posx[:-1]))  
        else:
            pos.iloc[i,0]=pos_.iloc[i-1,0] 
    return pos

        
        
        
        
    while any(dx>6):
    dx = np.concatenate(([0],pos_x[1:]-pos_x[:-1]))
        for i in range(len(pos_)):
            if dx[i] > 6:
                pos.iloc[i,0]=pos_.iloc[i-1,0]
            else:
                pos.iloc[i,0]=pos_.iloc[i-1,0] 
    posx=array(pos_['X'])
    dx=np.concatenate(([0],posx[1:]-posx[:-1]))   
    return pos        
    
    
computeDist(pos_)
        
        
        
        
        


