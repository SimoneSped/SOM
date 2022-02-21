# -*- coding: utf-8 -*-

import numpy as np
from line_feature import LineFeature


# TODO: add broadening and HFS?


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

    def add_features(self):
        self.centroids = np.random.randint(self.size, size=self.num_features)
        FWHMs = np.random.uniform(
            3, 8, self.num_features) * 2.3548
        max_intensities = np.random.uniform(55, 255, self.num_features)
        for i in range(0, self.num_features):
            self.intensities += LineFeature(
                self.centroids[i],
                FWHMs[i],
                max_intensities[i],
                self.lambda_range).intensities

    def add_noise(self):
        white_gaussian_noise = np.random.normal(0, 1, self.size)
        pwr_signal = np.sqrt(np.sum(self.intensities**2))/self.size
        pwr_noise = np.sqrt(np.sum(white_gaussian_noise**2))/self.size

        scale_factor = (pwr_signal / pwr_noise) / self.target_snr
        white_gaussian_noise = scale_factor * white_gaussian_noise

        self.intensities += white_gaussian_noise
