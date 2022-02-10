# -*- coding: utf-8 -*-

import numpy as np
from line_feature import GaussianFeature

# TODO: add noise
# TODO: add broadening?
# TODO: general fine-tuning of parameters

class SyntheticSpectrum:

    def __init__(self, interval=None, step=0.1, num_features=5):
        """
        Class for the creation of synthetic spectra
        :type interval: numpy array
        :type step: float
        :type num_features: int
        """
        if interval is None:
            self._interval = [0, 10000]
        else:
            self._interval = interval
        self._lambda_range = np.arange(interval[0], interval[1], step)
        self._step = step
        self._num_features = num_features
        self._intensities = np.zeros(shape=(len(self._lambda_range)))
        self.add_features()

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        self._interval = value
        self._lambda_range = np.arange(self._interval[0], self._interval[1], self._step)

    @property
    def lambda_range(self):
        return self._lambda_range

    @lambda_range.setter
    def lambda_range(self, value):
        self._lambda_range = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value
        self._lambda_range = np.arange(self._interval[0], self._interval[1], self._step)

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

    def add_features(self):
        centroids = np.random.randint(self._interval[1], size=self._num_features) + self._interval[0]
        FWHMs = np.random.uniform(
            20*round(self._interval[1]/10000), 255*round(self._interval[1]/10000), self._num_features)
        max_int = np.random.uniform(1, 255, self._num_features)
        for i in range(0, self._num_features):
            self._intensities += GaussianFeature(centroids[i], FWHMs[i], max_int[i], self._lambda_range).intensities
