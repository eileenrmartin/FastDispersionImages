import math
import numpy as np
import scipy.fftpack as ft
import traceClass as tr


########## calculate sigma, the common factor in all dispersion images ##########


def sigma3D(traces, xvelocities, yvelocities, minFreq, maxFreq, filterFunction):
	'''traces is a list of trace objects (defined in traceClass.py) assumed to all have the same length traces with the same sampling rate. velocities are a 1D numpy array of velocities of interest (m/s) for both the x and y direction. minFreq and maxFreq are the minimum/maximum positive frequencies of interest (Hz). filterFunction is a user-defined function that takes a single trace as input and filters it. This calculates the common factor (sigma) in all dispersion images that use these traces as receivers, which is returned as a 2D numpy array nxVel x nyVel x 2*nFrq where nFrq is the number of frequency bins between minFreq and maxFreq. This will return results in the order velocities[0],velocities[1],...,velocities[-1] and in the other direction minFreq,...,maxFreq,-maxFreq,...,-minFreq'''

	# define dimensions of sigma
	nxVel = xvelocities.size
	nyVel = yvelocities.size
	minFrqIdx = traces[0].getIdxFromHz(minFreq)
	maxFrqIdx = traces[0].getIdxFromHz(maxFreq)
	nFrq = maxFrqIdx-minFrqIdx
	# initialize the common factor to all dispersion images
	sigma = np.zeros((nxVel,nyVel,2*nFrq))

	# calculate the frequencies of interest
	actualMinFrq = minFrqIdx*traces[0].getNHzPerBin()
	actualMaxFrq = maxFrqIdx*traces[0].getNHzPerBin()
	posNegFrqs = np.hstack((np.linspace(actualMinFrq,actualMaxFrq,nFrq),-1*np.flipud(np.linspace(actualMinFrq,actualMaxFrq,nFrq))))

	# define the phase shift matrix
	px = 1.0/xvelocities # slowness vector
	py = 1.0/yvelocities
	phaseShiftMatX = np.tile(np.reshape(np.exp(2*np.pi*1j*np.outer(px,posNegFrqs)),(nxVel,1,2*nFrq)),(1,nyVel,1))
	phaseShiftMatY = np.tile(np.exp(2*np.pi*1j*np.outer(py,posNegFrqs)),(nxVel,1,1))

	# for each trace, apply a phase shift to 
	for r in traces:

		filterFunction(r) # call user-defined filters

		# just get the spectrum limited to frequencies of interest
		if(minFrqIdx <= 1):
			subsetSpec = np.hstack((r.dataSpec[minFrqIdx:maxFrqIdx],r.dataSpec[1-maxFrqIdx:]))
		else:
			subsetSpec = np.hstack((r.dataSpec[minFrqIdx:maxFrqIdx],r.dataSpec[1-maxFrqIdx:1-minFrqIdx])) 

		# add phase shifted subset of fourier transform of filtered data to sigma
		sigma = sigma + np.tile(subsetSpec,(nxVel,nyVel,1))*np.power(phaseShiftMatX,r.x)*np.power(phaseShiftMatY,r.y)

	# return an nxVel x nyVel x 2*nFrq array
	return sigma



############## calculate dispersion image with one virtual source ##############


def dispImg3D(aFilteredTrace, xvelocities, yvelocities, minFreq, maxFreq, sigma):
	'''aFilteredTrace is a trace object (defined in traceClass.py) assumed to have already been filtered that will act as a virtual source for this dispersion image. minFreq and maxFreq are the minimum/maximum positive frequencies of interest (Hz). velocities are a 1D numpy array of velocities of interest in the x and y directions (m/s). minFreq and maxFreq are the minimum/maximum positive frequencies of interest (Hz). sigma is the common factor in all dispersion images that use these traces as receivers, which is returned by sigma2D() as a 2D numpy array nxVel x nyVel x 2*nFrq where nFrq is the number of frequency bins between minFreq and maxFreq. The sigma matrix will be in the order velocities[0],velocities[1],...,velocities[-1] and in the other direction minFreq,...,maxFreq,-maxFreq,...,-minFreq. This function will return a 3D dispersion image in the order velocities[0],velocities[1],...,velocities[-1] in the first and second dimensions and in the third direction minFreq,...,maxFreq. It will have been symmetrized for positive and negative frequencies, and all returned values will be non-negative.'''

	# define dimensions of spectrum of interest
	minFrqIdx = aFilteredTrace.getIdxFromHz(minFreq)
	maxFrqIdx = aFilteredTrace.getIdxFromHz(maxFreq)
	nFrq = maxFrqIdx - minFrqIdx

	# just get the data spectrum limited to frequencies of interest
	if(minFrqIdx <= 1):
		subsetSpec = np.hstack((aFilteredTrace.dataSpec[minFrqIdx:maxFrqIdx],aFilteredTrace.dataSpec[1-maxFrqIdx:]))
	else:
		subsetSpec = np.hstack((aFilteredTrace.dataSpec[minFrqIdx:maxFrqIdx],aFilteredTrace.dataSpec[1-maxFrqIdx:1-minFrqIdx])) 

	# calculate the frequencies of interest
	actualMinFrq = minFrqIdx*aFilteredTrace.getNHzPerBin()
	actualMaxFrq = maxFrqIdx*aFilteredTrace.getNHzPerBin()
	posNegFrqs = np.hstack((np.linspace(actualMinFrq,actualMaxFrq,nFrq),-1*np.flipud(np.linspace(actualMinFrq,actualMaxFrq,nFrq))))

	# define the phase shift matrix
	px = 1.0/xvelocities #slowness vector
	py = 1.0/yvelocities # slowness vector
	nxVel = xvelocities.size
	nyVel = yvelocities.size
	phaseShiftMatX = np.tile(np.reshape(np.exp(-2*np.pi*1j*np.outer(px,posNegFrqs)),(nxVel,1,2*nFrq)),(1,nyVel,1))
	phaseShiftMatY = np.tile(np.exp(-2*np.pi*1j*np.outer(py,posNegFrqs)),(nxVel,1,1))

	# calculate the dispersion image my multiplying data spectrum by phase shifts
	dispImgPosNeg = np.tile(np.conj(subsetSpec),(nxVel,nyVel,1))*np.power(phaseShiftMatX,aFilteredTrace.x)*np.power(phaseShiftMatY,aFilteredTrace.y)*sigma

	# symmetrize positive and negative dispersion images
	dispImg = np.absolute(dispImgPosNeg[:,:,:nFrq]) + np.absolute(np.fliplr(dispImgPosNeg[:,:,nFrq:]))

	return dispImg




############ calculate a stack of dispersion images ########################

def dispImgStack3D(traces, xvelocities, yvelocities, minFreq, maxFreq, filterFunction):
	'''traces is a list of trace objects (defined in traceClass.py) assumed to all have the same length traces with the same sampling rate. velocities are a 1D numpy array of x and y velocities of interest (m/s). minFreq and maxFreq are the minimum/maximum positive frequencies of interest (Hz). filterFunction is a user-defined function that takes a single trace as input and filters it. This function will return a 3D dispersion image in the order velocities[0],velocities[1],...,velocities[-1] in the first and second dimensions and in the third direction minFreq,...,maxFreq. It will have been symmetrized for positive and negative frequencies, and all returned values will be non-negative.'''

	# calculate the sigma common factor to all dispersion images
	sigmaFactor = sigma3D(traces, xvelocities, yvelocities, minFreq, maxFreq, filterFunction)

	# set up zero stack 
	nxVel = xvelocities.size
	nyVel = yvelocities.size 
	nFrq = sigmaFactor.shape[2]/2
	dispImgStack = np.zeros((nxVel,nyVel,nFrq))

	# for each virtual source calculate the dispersion image and add it to the stack
	for virtualSource in traces:
		dispImgStack += dispImg3D(virtualSource, xvelocities, yvelocities, minFreq, maxFreq, sigmaFactor)

	return dispImgStack