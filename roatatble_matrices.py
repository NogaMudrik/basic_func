# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:49:25 2024

@author: noga mudrik
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation

m = rand(3,3) # m is an array of (x,y,z) coordinate triplets

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(m)): # plot each point + it's index as text above
  x = m[i,0]
  y = m[i,1]
  z = m[i,2]
  label = i
  ax.scatter(x, y, z, color='b')
  ax.text(x, y, z, '%s' % (label), size=20, zorder=1, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def animate(frame):
  ax.view_init(30, frame/4)
  plt.pause(.001)
  return fig

anim = animation.FuncAnimation(fig, animate, frames=200, interval=50)
HTML(anim.to_html5_video())