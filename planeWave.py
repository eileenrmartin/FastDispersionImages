#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:35:38 2021

@author: anu
"""
import numpy as np
import matplotlib.pyplot as plt

def planeWave(k, w, theta, sensors, totTime):
    
    '''
    inputs:
        k = wave number array, dimensions = 2x1 
        w = angular frequency (2*pi*f), float
        theta = angle in radians, float
        sensors = number of sensors spaced 2m apart, int
        totTime = total time in seconds, float    
        
    output:
        wave = plane wave array, dimensions = (sensors)x(4*totTime*w/pi)
    '''
    
    x      = np.zeros((sensors, 2))
    x[:,0] = np.linspace(0, 2*sensors, sensors)
    t      = np.linspace(0, totTime, 4*int(w/np.pi)*totTime)
    wave   = np.zeros((len(x), len(t)))
        
    for i in range(len(x)): # loop over rows (channel #)
        wave[int(i),:] = np.cos(np.matmul(np.transpose(x[int(i)]), k)- w*t)

    return wave
              
## create single frequency plane wave test case:

w       = 2*np.pi*5 # frequency
v       = 350 # speed of sound underground: 150 - 700 m/s
theta   = np.pi/6 # angle
sensors = 20 # number of sensors, assume 2m distance between each
totTime = 10 # total time in seconds

# save x and y component of wave number
k    = w/(v*2*np.pi)      
k_x  = np.abs(k)*np.cos(theta)
k_y  = np.abs(k)*np.sin(theta)
kArr = np.array((k_x, k_y))

wave = planeWave(kArr, w, theta, sensors, totTime) 

plt.imshow(wave, aspect='auto', cmap=plt.cm.get_cmap('RdBu_r'))
plt.colorbar()
plt.xlabel('time')
plt.ylabel('channel')
