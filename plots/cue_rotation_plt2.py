import matplotlib.pyplot as plt


standard_ang=all_circMean.iloc[:,0]

rot_ang=[]
for rots in [45,90,180]:
    if rots ==cond3:
        for i in range(len(all_circMean)):
            rot_ang.append((((abs(all_circMean.iloc[i,1]-all_circMean.iloc[i,0]))))-deg2rad(rots)+ (all_circMean.iloc[i,0]))
            
plt.figure()
    
plt.scatter(standard_ang,rot_ang, c='r')

plt.title('Cue Card Rotation')


gca().set_ylabel('Observed Mean PFD (rad)')
gca().set_xlabel('Expected Mean PFD (rad)')

gca().set_ylim(0,2*np.pi)
gca().set_xlim(0,2*np.pi)



plt.figure()
hist(standard_ang,bins=5)

new=all_circMean.iloc[:,1]-deg2rad(180)
hist(new,bins=5,color='red')

gca().set_xlim([-2*pi,2*pi])
