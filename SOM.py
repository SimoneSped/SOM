# -*- coding: utf-8 -*-
import numpy as np
from neuron import Neuron
from cluster import Cluster
import random


class SOM:

    def __init__(self, x_size=20, y_size=20, size_neurons=10000, learning_rate_0=0.5, radius_0=0.1,
                 cluster_distance_threshold=0.04, input_data=None):
        """
        Class for the implementation of the self-organizing maps
        :type x_size: int
        :type y_size: int
        :type size_neurons: int
        :type learning_rate_0: float between 0 and 1
        :type radius_0: float between 0 and 1
        :type cluster_distance_threshold? float between 0 and 1
        :type input_data: numpy array
        """
        self.x_size = x_size
        self.y_size = y_size
        self.size_neurons = size_neurons
        self.iteration = 0
        self.time_constant = 100
        self.learning_rate_0 = learning_rate_0
        self.learning_rate = learning_rate_0
        self.radius_0 = radius_0
        self.radius = radius_0
        self.cluster_distance_threshold = cluster_distance_threshold
        self.input_data = input_data
        self.neuron_map = np.empty(
            shape=(x_size, y_size),
            dtype=object
        )
        self.clusters = np.array(
            [],
            dtype=object
        )
        for i in range(self._x_size):
            for j in range(self._y_size):
                self._neuron_map[i][j] = Neuron(i / x_size, j / y_size, np.random.uniform(0.001, 1, size_neurons))

    @property
    def x_size(self):
        return self._x_size

    @x_size.setter
    def x_size(self, value):
        self._x_size = value

    @property
    def y_size(self):
        return self._y_size

    @y_size.setter
    def y_size(self, value):
        self._y_size = value

    @property
    def size_neurons(self):
        return self._size_neurons

    @size_neurons.setter
    def size_neurons(self, value):
        self._size_neurons = value

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value

    @property
    def time_constant(self):
        return self._time_constant

    @time_constant.setter
    def time_constant(self, value):
        self._time_constant = value

    @property
    def learning_rate_0(self):
        return self._learning_rate_0

    @learning_rate_0.setter
    def learning_rate_0(self, value):
        self._learning_rate_0 = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def radius_0(self):
        return self._radius_0

    @radius_0.setter
    def radius_0(self, value):
        self._radius_0 = value

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @property
    def cluster_distance_threshold(self):
        return self._cluster_distance_threshold

    @cluster_distance_threshold.setter
    def cluster_distance_threshold(self, value):
        if value < 0 or value > 1:
            raise ValueError
        else:
            self._cluster_distance_threshold = value

    @property
    def input_data(self):
        return self._input_data

    @input_data.setter
    def input_data(self, value):
        self._input_data = value

    @property
    def neuron_map(self):
        return self._neuron_map

    @neuron_map.setter
    def neuron_map(self, value):
        self._neuron_map = value

    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, value):
        self._clusters = value

    def find_bmu(self, input_vector):
        # compute euclidian distances from the input vector to the weight vector of the neurons
        distances = np.array([np.linalg.norm(self.neuron_map[i][j].weights - input_vector)
                              for i in range(self.x_size)
                              for j in range(self.y_size)]).reshape((self.x_size, self.y_size))
        # return the index of the neuron with minimal distance (a.k.a. best-matching unit)
        minimal_distance = np.where(distances == np.amin(distances))
        return [minimal_distance[0][0], minimal_distance[1][0]]

    def update_grid(self, input_vector):
        # TODO: optimize this loop
        # find the best-matching unit
        bmu_x_index, bmu_y_index = self.find_bmu(input_vector)
        bmu = self.neuron_map[bmu_x_index][bmu_y_index]
        for neuron_line in self.neuron_map:
            for neuron in neuron_line:
                # find each neuron that falls into the radius from the bmu at this iteration
                if (neuron.x - bmu.x) ** 2 + (neuron.y - bmu.y) ** 2 <= self.radius ** 2:
                    # update weights of the found neurons accordingly
                    neuron.weights = neuron.weights + self.learning_rate * (
                            input_vector - neuron.weights)

                    # update positions of the found neurons accordingly
                    neuron.x += self.learning_rate * (
                            bmu.x - neuron.x)
                    neuron.y += self.learning_rate * (
                            bmu.y - neuron.y)
        self.update_learning_rate()
        self.update_radius()
        self.iteration = self.iteration + 1

    def update_radius(self):
        self.radius = self.radius_0 * np.exp(-self.iteration / self.time_constant)

    def update_learning_rate(self):
        self.learning_rate = self.learning_rate_0 * np.exp(-self.iteration / self.time_constant)

    def find_clusters(self):
        # FoF

        # make list of valid points
        list_points = [
            [i, j]
            for i in range(self.x_size)
            for j in range(self.y_size)
        ]

        count = 0
        while count < self.x_size * self.y_size:
            # choose random valid point to start with
            idx = random.randint(0, len(list_points) - 1)
            start_point = list_points.pop(idx)
            start_neuron = self.neuron_map[start_point[0], start_point[1]]
            cluster = Cluster([start_neuron], self.cluster_distance_threshold)
            count += 1
            for point in list_points:
                # calculate distance for each point to the starting neuron
                distance = np.sqrt((self.neuron_map[point[0]][point[1]].x - start_neuron.x) ** 2 + (
                        self.neuron_map[point[0]][point[1]].y - start_neuron.y) ** 2)
                if distance <= cluster.distance_threshold:
                    # add member to cluster
                    cluster.add_member(self.neuron_map[point[0]][point[1]], distance)
                    # increase total count
                    count += 1
                    # remove indexes from list of valid points
                    list_points.remove(point)

            # repeat for each of the friends
            for j in range(1, len(cluster.members)):
                for point in list_points:
                    # calculate distance for each remaining point to the friends of the starting neuron
                    distance = np.sqrt((self.neuron_map[point[0]][point[1]].x - cluster.members[j].x) ** 2 + (
                            self.neuron_map[point[0]][point[1]].y - cluster.members[j].y) ** 2)
                    if distance <= cluster.distance_threshold:
                        # add member to cluster
                        cluster.add_member(self.neuron_map[point[0]][point[1]], distance)
                        # increase total count
                        count += 1
                        # remove indexes from list of valid points
                        list_points.remove(point)
            # more or less subjective threshold for number of members
            #if len(cluster.members) > 3:
            self.clusters = np.append(self.clusters, cluster)
            # take new random point, but do not iterate over the previously found friends

        # sort according to the higher clustering index
        self.clusters = sorted(self.clusters, key=lambda n: n.clustering_index)[::-1]

    def match_input_to_cluster(self):
        pass

    def start(self):
        [self.update_grid(vector) for vector in self.input_data]
        self.find_clusters()
        self.match_input_to_cluster()
