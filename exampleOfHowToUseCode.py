########### Templates of how to use all 2D and 3D functions in fastDispImg2D
########### and fastDispImg3D.
########### Author: Eileen R. Martin, Stanford University, Dec. 2015
########### Available at github.com/eileenrmartin/FastDispersionImages
########### Last edited 3/28/2021 by Anu Trivedi

import traceClass as tr
import numpy as np
import fastDispImg2D as fdi2
import fastDispImg3D as fdi3
import matplotlib.pyplot as plt

######### example filter functions ##########

def myFilterFunc(someTrace):
	''' This is an example filter function that only scales data by a factor of two'''
	someTrace.scale_data(2.0)
	##### really you should call your own filters here ######
	##### for example temporal or frequency whitening  ######


def passFilterFunc(someTrace):
	''' This is a filter function that does nothing to a trace object '''
	pass



################### example using 2D code #############################


# fill in your data file name here and read 
datapath2D='/specify/the/path/to/some/data/file'

# list of the receiver incices you will read
receiverList2D = ###########

# fill in your data sample rate here (it may come from reading your data file)
dt2D = ##### fill in your sample rate in seconds here (float)

# fill in velocities of interest
minVel = ###### fill in your minimum velocity in m/s (float)
maxVel = ###### fill in your maximum velocity in m/s (float)
nVel = ##### fill in the desired number of velocities (int)
velocities = np.linspace(minVel,maxVel,nVel)

# fill in frequencies of interest
minFrq2D = ###### fill in minimum positive frequency of interest in Hz (float)
maxFrq2D = ###### fill in maximum positive frequency of interest in Hz (float)

# read in the receivers of interest
receivers2D = []
for r in receiverList2D:
	thisData = ###### call a function that reads time series of receiver number r
	position = ####### call a function that reads the position of receiver number r in meters
	thisReceiver = tr.trace(thisData,dt2D,position) # create trace object
	receivers2D.append(thisReceiver) # append the trace object to the list

print("Length of receiver list: ", len(receivers2D))

# Calculate a stack of dispersion images over all virtual sources
dispImgStack2D = fdi2.dispImgStack2D(receivers2D, velocities, minFrq2D, maxFrq2D, myFilterFunc)
print('dispersion images stacked over all virtual sources')
print(dispImgStack2D)
print(dispImgStack2D.shape)

# just calculate the common factor sigma of all dispersion images
# Use the passFilterFunction here because traces were already filtered in calculating sigma2D
sigma2D = fdi2.sigma2D(receivers, velocities, minFrq2D, maxFrq2D, passFilterFunc)
print('sigma 2D')
print(sigma2D)
print(sigma2D.shape)

# using the sigma factor above, calculate a dispersion image from virtual source 0
singleDispImg2D = fdi2.dispImg2D(receivers2D[0], velocities, minFrq2D, maxFrq2D, sigma2D)
print('dispersion image with first receiver as virtual source')
print(singleDispImg2D)
print("Dispersion image dimesnsions: ", singleDispImg2D.shape)


# uncomment to plot:
# plt.imshow(singleDispImg2D, aspect='auto')
# plt.ylabel('velocity (m/s)')
# plt.xlabel('frequency (Hz)')




############# example using 3D code ############################
# fill in your data file name here 
datapath3D='/specify/the/path/to/some/data/file'

# list of the receiver incices you will read
receiverList3D = ###########

# fill in your data sample rate here (it may come from reading your data file)
dt3D = ##### fill in your sample rate in seconds here (float)

# fill in velocities of interest
minxVel = ###### fill in your minimum velocity in m/s (float)
maxxVel = ###### fill in your maximum velocity in m/s (float)
nxVel = ##### fill in the number of x velocities of interest
xvelocities = np.linspace(minxVel,maxxVel,nxVel)

# fill in velocities of interest
minyVel = ###### fill in your minimum velocity in m/s (float)
maxyVel = ###### fill in your maximum velocity in m/s (float)
nyVel = ##### fill in the number of y velocities of interest
yvelocities = np.linspace(minyVel,maxyVel,nyVel)

# fill in frequencies of interest
minFrq3D = ###### fill in minimum positive frequency of interest in Hz (float)
maxFrq3D = ###### fill in maximum positive frequency of interest in Hz (float)

# read in the receivers of interest
receivers3D = []
for r in receiverList3D:
	thisData = ###### call a function that reads time series of receiver number r
	positionx = ####### call a function that reads the x position of receiver number r in meters
	positiony = ####### call a function that reads the y position of receiver number r in meters
	thisReceiver = tr.trace(thisData,dt3D,positionx,positiony) # create trace object
	receivers3D.append(thisReceiver) # append the trace object to the list

# Calculate a stack of dispersion images over all virtual sources
dispImgStack3D = fdi3.dispImgStack3D(receivers3D, xvelocities, yvelocities, minFrq3D, maxFrq3D, myFilterFunc)
print('dispersion images stacked over all virtual sources')
print(dispImgStack3D)
print(dispImgStack3D.shape)

# just calculate the common factor sigma of all dispersion images
# Use the passFilterFunction here because traces were already filtered when calculating the stack
sigma3D = fdi3.sigma3D(receivers3D, xvelocities, yvelocities, minFrq3D, maxFrq3D, passFilterFunc)
print('3d sigma')
print(sigma3D)
print(sigma3D.shape)

# using the sigma factor above, calculate a dispersion image from virtual source 0
singleDispImg3D = fdi3.dispImg3D(receivers3D[0], xvelocities, yvelocities, minFrq3D, maxFrq3D, sigma3D)
print('dispersion image with first receiver as virtual source')
print(singleDispImg3D)
print(singleDispImg3D.shape)


