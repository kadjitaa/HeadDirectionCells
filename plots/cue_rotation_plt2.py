import matplotlib.pyplot as plt

plt.figure()






pfd2=rad2deg(all_pfd.iloc[:,1].astype('int64'))
for rots in [45,90,180]:
    if rots ==cond3:
        pfd2_expected=abs(rad2deg((all_pfd.iloc[:,0].astype('int64'))+3.14)) % 2*np.pi


gca().set_ylabel('Observed PFD')
gca().set_xlabel('Expected PFD')

plt.scatter(pfd2,pfd2_expected, c='red')
plt.title('Cue Card Rotation')

gca().set_ylim([0,360])
gca().set_xlim([0,360])

gca().set_ylabel('Observed PFD')
gca().set_xlabel('Expected PFD')