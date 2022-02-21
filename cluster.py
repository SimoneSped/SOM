# -*- coding: utf-8 -*-
import numpy as np


class Cluster:

    def __init__(self, members=np.empty(shape=1), distance_threshold=0.01):
        """
        Class for the implementation of the cluster in a FoF sense
        :type members: numpy array of neurons
        :type distance_threshold: float
        """
        self.members = members
        self.clustering_index = 0
        self.distance_threshold = distance_threshold

    @property
    def members(self):
        return self._members

    @members.setter
    def members(self, value):
        self._members = value

    @property
    def clustering_index(self):
        return self._clustering_index

    @clustering_index.setter
    def clustering_index(self, value):
        if value < 0:
            raise ValueError
        else:
            self._clustering_index = value

    @property
    def distance_threshold(self):
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, value):
        if value <= 0:
            raise ValueError
        else:
            self._distance_threshold = value

    def update_clustering_index(self, distance):
        self.clustering_index = len(self.members)*(1/(len(self.members)-1)*self.clustering_index + 1/distance)

    def add_member(self, new_member, distance):
        self.members = np.append(self.members, new_member)
        self.update_clustering_index(distance)
