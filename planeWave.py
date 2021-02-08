#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:35:38 2021

@author: anu
"""
import numpy as np
import matplotlib.pyplot as plt

def planeWave(k, w, theta, sensors, totTime):
    
    # make x and k vectors for future cases (add this in)
    
    x    = np.linspace(0, 2*sensors, sensors)
    t    = np.linspace(0, totTime, 4*int(w/np.pi)*totTime)
    wave = np.zeros((len(x), len(t)))
    
    
    for i in range(len(x)): # loop over rows (channel #)
        wave[int(i),:] = np.cos(x[int(i)]*k - w*t)

    return wave
              
## create single frequency plane wave test case:


w = 2*np.pi*10 # frequency
v = 350 # speed of sound underground: 150 - 700 m/s
k = w/(v*2*np.pi) 
theta = 0 # angle
sensors = 50 # number of sensors, assume 2m distance between each
totTime = 10 # total time in seconds
    
k_x = np.abs(k)*np.cos(theta)
# add in k_y term

wave = planeWave(k_x, w, theta, sensors, totTime) 

plt.imshow(wave, aspect='auto', cmap=plt.cm.get_cmap('RdBu_r'))
plt.colorbar()
plt.xlabel('time')
plt.ylabel('channel')
