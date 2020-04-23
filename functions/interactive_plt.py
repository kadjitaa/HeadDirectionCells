# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:18:25 2020

@author: kasum
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
pos=pd.read_excel(r'C:\Users\kasum\Downloads\M2-c2.xlsx',nrows=500)

sec=0.01
fig, ax = plt.subplots()

x = np.array(position['x'])#np.arange(0, 2*np.pi, 0.01)
y=np.array(position['z'])

[line]=ax.plot(x,y,lw=3, color='r')

def init():  # only required for blitting to give a clean slate.
    line.set_data([],[])
    plt.savefig('figure0.png')
    plt.pause(sec)
    return ([line])


def animate(i):
    line.set_data(x[:i],y[:i])# update the data.
    filename='figure'+str(i)+'.png'
    plt.savefig(filename)
    plt.pause(sec)
    return ([line])


animation = anim.FuncAnimation(
    fig, animate, init_func=init, frames=len(x)+1,interval=100, blit=False)


#Save
# Save Files
animation.save('animated_gif.gif',writer=anim.PillowWriter(fps=300))

#FFwriter = anim.FFMpegWriter(fps=60)
#animation.save('animated_gif.mp4',writer=anim.FFMpegWriter(fps=60))
#anim.FileMovieWriter(fps=60)
#animation.save('animated_avi.avi',writer=anim.FFMpegWriter(fps=1))
#animation.save('animated_mkv.mkv',writer=anim.FFMpegWriter(fps=1))
# Set up formatting for the movie files
Writer = anim.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
animation.save('animated_gif.mp4',writer=writer)

