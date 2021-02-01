########### Basics of a trace class that users can build on.
########### Author: Eileen R. Martin, Stanford University, Dec. 2015
########### Available at github.com/eileenrmartin/FastDispersionImages

import numpy as np
import scipy.fftpack as ft

class trace:


	def __init__(self, data, dt, x, y=0):
		'''data should be a 1d numpy array of floats representing a time series recorded at this receiver, dt should be seconds between samples in data (float), x is the x-position of this receiver in meters (float), y is the y-position of this receiver in meters (float). When dealing with a linear array, only include the x value.'''
		self.data = data # time series data 
		self.nSamples = data.size # length of data
		self.dt = dt # time (s) between samples
		self.x = x # x-position (m) of receiver
		self.y = y # y-position (m) of receiver
		self.dataSpec = None # will hold the spectrum
		self.set_dataSpec() # set the spectrum


	def set_dataSpec(self):
		'''Take an FFT of self.data and scale by number of samples'''
		self.dataSpec = ft.fft(self.data)
		self.dataSpec /= self.nSamples

	def getNHzPerBin(self):
		'''Get the number of Hz falling into each frequency bin in dataSpec'''
		NyquistFrq = 0.5/self.dt # Nyquist frequency (Hz)
		return NyquistFrq/(self.nSamples/2)

	def getIdxFromHz(self, freqHz):
		'''Get the index in self.dataSpec of the positive frequency (Hz) specified by freqHz'''
		NyquistFrq = 0.5/self.dt # Nyquist frequency (Hz)
		if(freqHz > NyquistFrq):
			print("Error: requested frequency outside range.")
			return 0.5 # fraction index should cause errors
		nHzPerBin = self.getNHzPerBin() # number of Hz in each bin of dataSpec

		idx = int(freqHz/nHzPerBin) 
		return idx


	#### Here's an example of a filter, but you ####
	#### should add more filters. Things like   ####
	#### temporal normalization, spectral       ####
	#### whitening, bandpass filters are standard ##
	#### types of filters you could add here    ####
	#### in this class definition.              ####
	def scale_data(self,c):
		'''Scale the data and the data spectrum by multiplying by c (a scalar float)'''
		self.data = self.data*c
		self.dataSpec = self.dataSpec*c

