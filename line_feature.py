# -*- coding: utf-8 -*-
import numpy as np


class GaussianFeature:
    
    def __init__(self, centroid, FWHM, max_intensity, spectrum_range):
        """
        Class for the implementation of the single gaussian features
        :type centroid: float
        :type FWHM: float
        :type max_intensity: float
        :type spectrum_range: numpy array
        """
        self.centroid = centroid
        self.FWHM = FWHM
        self.max_intensity = max_intensity
        self.spectrum_range = spectrum_range
        self.sigma = FWHM/2.3548
        self.intensities = self.profile()
    
    @property
    def centroid(self):
        return self._centroid
    
    @centroid.setter
    def centroid(self, value):
        if value < 0:
            raise ValueError
        else:
            self._centroid = value
            
    @property
    def FWHM(self):
        return self._FWHM
    
    @FWHM.setter
    def FWHM(self, value):
        if value <= 0:
            raise ValueError
        else:
            self._FWHM = value
            self._sigma = self._FWHM/2.3548

    @property
    def max_intensity(self):
        return self._max_intensity

    @max_intensity.setter
    def max_intensity(self, value):
        if value <= 0:
            raise ValueError
        else:
            self._max_intensity = value
            
    @property
    def spectrum_range(self):
        return self._spectrum_range
    
    @spectrum_range.setter
    def spectrum_range(self, value):
        if value[0] < 0:
            raise ValueError
        else:
            self._spectrum_range = value
    
    @property
    def intensities(self):
        return self._intensities
    
    @intensities.setter
    def intensities(self, value):
        self._intensities = value

    def profile(self):
        return self.max_intensity*np.exp(-0.5*(((self.spectrum_range-self.centroid)/self.sigma)**2))
