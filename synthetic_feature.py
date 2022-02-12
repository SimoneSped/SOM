# -*- coding: utf-8 -*-

import numpy as np
from line_feature import GaussianFeature

# TODO: add broadening?
# TODO: general fine-tuning of parameters


class SyntheticSpectrum:

    def __init__(self, size=10000, num_features=5, noise_scale=0.1):
        """
        Class for the creation of synthetic spectra
        :type size: int
        :type num_features: int
        :type noise_scale: float
        """
        self.size = size
        self.lambda_range = np.arange(0, size, 1)
        self.num_features = num_features
        self.intensities = np.zeros(shape=size)
        self.noise_scale = noise_scale
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
    def noise_scale(self):
        return self._noise_scale

    @noise_scale.setter
    def noise_scale(self, value):
        self._noise_scale = value

    def add_features(self):
        centroids = np.random.randint(self.size, size=self.num_features)
        FWHMs = np.random.uniform(
            20, 50, self.num_features)
        max_int = np.random.uniform(0, 255, self.num_features)
        for i in range(0, self.num_features):
            self.intensities += GaussianFeature(centroids[i], FWHMs[i], max_int[i], self.lambda_range).intensities

    def add_noise(self):
        # Method 1
        # self.intensities += np.random.normal(0, 1, self.interval[1]) * self.intensities * self.noise_scale

        # Method 2 (mask signal with noise)

        self.intensities += self.noise_scale * np.random.normal(0, 1, self.size) * np.sqrt(np.amax(np.abs(self.intensities)))
