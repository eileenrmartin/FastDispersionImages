#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:35:38 2021

@author: anu
"""
import numpy as np
import matplotlib.pyplot as plt

def planeWave(k, w, theta, sensors, totTime):
    
    x    = np.linspace(0, 2*sensors, sensors)
    t    = np.linspace(0, totTime, w*totTime)
    wave = np.zeros((len(x), len(t)))
    
    for i in range(len(x)): # loop over rows (channel #)
        for j in range(len(t)): # loop over columns (time)
            wave[int(i),int(j)] = np.cos(x[int(i)]*k - w*t[int(j)])

    return wave
              
## create single frequency plane wave test case:

k = 4 # wave number (q: select randomly or calculate?)
w = 10 # frequency
theta = 30 # angle
sensors = 50 # number of sensors, assume 2m distance between each
totTime = 10 # total time in seconds
    
k = np.abs(k)*np.cos(theta) 

wave = planeWave(k, w, theta, sensors, totTime) 

plt.imshow(wave, cmap=plt.cm.get_cmap('RdBu_r'))
plt.colorbar()
plt.xlabel('time')
plt.ylabel('channel')
