########### Code for calculating 2D dispersion images quickly.
########### Author: Eileen R. Martin, Stanford University, Dec. 2015
########### Available at github.com/eileenrmartin/FastDispersionImages

import math
import numpy as np
import scipy.fftpack as ft
import traceClass as tr



def sigma2D(traces, velocities, minFreq, maxFreq, filterFunction):
	'''traces is a list of trace objects (defined in traceClass.py) assumed to all have the same length traces with the same sampling rate. velocities are a 1D numpy array of velocities of interest (m/s). minFreq and maxFreq are the minimum/maximum positive frequencies of interest (Hz). filterFunction is a user-defined function that takes a single trace as input and filters it. This calculates the common factor (sigma) in all dispersion images that use these traces as receivers, which is returned as a 2D numpy array nVel x 2*nFrq where nFrq is the number of frequency bins between minFreq and maxFreq. This will return results in the order velocities[0],velocities[1],...,velocities[-1] and in the other direction minFreq,...,maxFreq,-maxFreq,...,-minFreq'''

	# define dimensions of sigma
	nVel = velocities.size
	minFrqIdx = traces[0].getIdxFromHz(minFreq)
	maxFrqIdx = traces[0].getIdxFromHz(maxFreq)
	nFrq = maxFrqIdx-minFrqIdx
	# initialize the common factor to all dispersion images
	sigma = np.zeros((nVel,2*nFrq))

	# calculate the frequencies of interest
	actualMinFrq = minFrqIdx*traces[0].getNHzPerBin()
	actualMaxFrq = maxFrqIdx*traces[0].getNHzPerBin()
	posNegFrqs = np.hstack((np.linspace(actualMinFrq,actualMaxFrq,nFrq),-1*np.flipud(np.linspace(actualMinFrq,actualMaxFrq,nFrq))))

	# define the phase shift matrix
	p = 1.0/velocities # slowness vector
	phaseShiftMat = np.exp(2*np.pi*1j*np.outer(p,posNegFrqs))

	# for each trace, apply a phase shift to 
	for r in traces:

		filterFunction(r) # call user-defined filters

		# just get the spectrum limited to frequencies of interest
		if(minFrqIdx <= 1):
			subsetSpec = np.hstack((r.dataSpec[minFrqIdx:maxFrqIdx],r.dataSpec[1-maxFrqIdx:]))
		else:
			subsetSpec = np.hstack((r.dataSpec[minFrqIdx:maxFrqIdx],r.dataSpec[1-maxFrqIdx:1-minFrqIdx])) 

		# add phase shifted subset of fourier transform of filtered data to sigma
		sigma = sigma + np.tile(subsetSpec,(nVel,1))*np.power(phaseShiftMat,r.x)

	# return an nVel x 2*nFrq array
	return sigma



def dispImg2D(aFilteredTrace, velocities, minFreq, maxFreq, sigma):
	'''aFilteredTrace is a trace object (defined in traceClass.py) assumed to have already been filtered that will act as a virtual source for this dispersion image. minFreq and maxFreq are the minimum/maximum positive frequencies of interest (Hz). velocities are a 1D numpy array of velocities of interest (m/s). minFreq and maxFreq are the minimum/maximum positive frequencies of interest (Hz). sigma is the common factor in all dispersion images that use these traces as receivers, which is returned by sigma2D() as a 2D numpy array nVel x 2*nFrq where nFrq is the number of frequency bins between minFreq and maxFreq. The sigma matrix will be in the order velocities[0],velocities[1],...,velocities[-1] and in the other direction minFreq,...,maxFreq,-maxFreq,...,-minFreq. This function will return a dispersion image in the order velocities[0],velocities[1],...,velocities[-1] and in the other direction minFreq,...,maxFreq. It will have been symmetrized for positive and negative frequencies, and all returned values will be non-negative.'''

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
	p = 1.0/velocities #slowness vector
	nVel = velocities.size
	phaseShiftMat = np.exp(-2*np.pi*1j*np.outer(p,posNegFrqs))

	# calculate the dispersion image my multiplying data spectrum by phase shifts
	dispImgPosNeg = np.tile(np.conj(subsetSpec),(nVel,1))*np.power(phaseShiftMat,aFilteredTrace.x)*sigma

	# symmetrize positive and negative dispersion images
	dispImg = np.absolute(dispImgPosNeg[:,:nFrq]) + np.absolute(np.fliplr(dispImgPosNeg[:,nFrq:]))

	return dispImg


def dispImgStack2D(traces, velocities, minFreq, maxFreq, filterFunction):
	'''traces is a list of trace objects (defined in traceClass.py) assumed to all have the same length traces with the same sampling rate. velocities are a 1D numpy array of velocities of interest (m/s). minFreq and maxFreq are the minimum/maximum positive frequencies of interest (Hz). filterFunction is a user-defined function that takes a single trace as input and filters it. This function will return a dispersion image stacked over all virtual sources in traces. The elements will be in the order velocities[0],velocities[1],...,velocities[-1] and in the other direction minFreq,...,maxFreq. It will have been symmetrized for positive and negative frequencies, and all returned values will be non-negative.'''

	# calculate the sigma common factor to all dispersion images
	sigmaFactor = sigma2D(traces, velocities, minFreq, maxFreq, filterFunction)

	# set up zero stack 
	nVel = velocities.size
	nFrq = sigmaFactor.shape[1]/2
	dispImgStack = np.zeros((nVel,nFrq))

	# for each virtual source calculate the dispersion image and add it to the stack
	for virtualSource in traces:
		dispImgStack += dispImg2D(virtualSource, velocities, minFreq, maxFreq, sigmaFactor)

	return dispImgStack