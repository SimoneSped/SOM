# -*- coding: utf-8 -*-
import numpy as np

# TODO: add broadening and HFS?


class LineFeature:

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
        self.sigma = FWHM / 2.3548
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
            self._sigma = self._FWHM / 2.3548

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
        return self.max_intensity * np.exp(-0.5 * (((self.spectrum_range - self.centroid) / self.sigma) ** 2))


class SyntheticSpectrum:

    def __init__(self, size=200, num_features=5, target_snr=0.1):
        """
        Class for the creation of synthetic spectra
        :type size: int
        :type num_features: int
        :type target_snr: int
        """
        self.size = size
        self.lambda_range = np.arange(0, size, 1)
        self.num_features = num_features
        self.intensities = np.zeros(shape=size)
        self.target_snr = target_snr
        self.centroids = []
        self.FWHMs = []
        self.max_intensities = []
        self.add_features()
        self.add_noise()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value
        self._lambda_range = np.arange(0, self.size, 1)

    @property
    def lambda_range(self):
        return self._lambda_range

    @lambda_range.setter
    def lambda_range(self, value):
        self._lambda_range = value

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, value):
        if value <= 0:
            raise ValueError
        else:
            self._num_features = value

    @property
    def intensities(self):
        return self._intensities

    @intensities.setter
    def intensities(self, value):
        self._intensities = value

    @property
    def target_snr(self):
        return self._target_snr

    @target_snr.setter
    def target_snr(self, value):
        self._target_snr = value

    @property
    def centroids(self):
        return self._centroids

    @centroids.setter
    def centroids(self, value):
        self._centroids = value

    @property
    def FWHMs(self):
        return self._FWHMs

    @FWHMs.setter
    def FWHMs(self, value):
        self._FWHMs = value

    @property
    def max_intensities(self):
        return self._max_intensities

    @max_intensities.setter
    def max_intensities(self, value):
        self._max_intensities = value

    def add_features(self):
        self.centroids = np.random.randint(self.size, size=self.num_features)
        self.FWHMs = np.random.uniform(
            3, 8, self.num_features) * 2.35482
        self.max_intensities = np.random.uniform(55, 255, self.num_features)
        for i in range(0, self.num_features):
            self.intensities += LineFeature(
                self.centroids[i],
                self.FWHMs[i],
                self.max_intensities[i],
                self.lambda_range).intensities

    def add_noise(self):
        white_gaussian_noise = np.random.normal(0, 1, self.size)
        pwr_signal = np.sqrt(np.sum(self.intensities**2))/self.size
        pwr_noise = np.sqrt(np.sum(white_gaussian_noise**2))/self.size

        scale_factor = (pwr_signal / pwr_noise) / self.target_snr
        white_gaussian_noise = scale_factor * white_gaussian_noise

        self.intensities += white_gaussian_noise
