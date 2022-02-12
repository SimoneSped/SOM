# -*- coding: utf-8 -*-
import numpy as np
from neuron import Neuron


class SOM:

    def __init__(self, x_size=20, y_size=20, size_neurons=10000, learning_rate_0=0.5, radius_0=10, input_data=None):
        self.x_size = x_size
        self.y_size = y_size
        self.size_neurons = size_neurons
        self.iteration = 0
        self.time_constant = 100
        self.learning_rate_0 = learning_rate_0
        self.learning_rate = learning_rate_0
        self.radius_0 = radius_0
        self.radius = radius_0
        self.input_data = input_data
        self.neuron_map = np.empty(
            shape=(x_size, y_size),
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

    def find_BMU(self, input_vector):
        # TODO: optimize this loop
        distances = np.array([np.linalg.norm(self.neuron_map[i][j].weights - input_vector)
                              for i in range(self.x_size)
                              for j in range(self.y_size)])
        distances = distances.reshape((self.x_size, self.y_size))
        minimal_distance = np.where(distances == np.amin(distances))
        return [minimal_distance[0][0], minimal_distance[1][0]]

    def update_grid(self, input_vector):
        # TODO: optimize this loop
        BMU_x_index, BMU_y_index = self.find_BMU(input_vector)
        BMU = self.neuron_map[BMU_x_index][BMU_y_index]
        for neuron_line in self.neuron_map:
            # none_found = False
            # none_found = 0
            for neuron in neuron_line:
                if (neuron.x - BMU.x) ** 2 + (neuron.y - BMU.y) ** 2 <= self.radius ** 2:
                    # one_found = True
                    # update weights
                    neuron.weights = neuron.weights + self.learning_rate * (
                            input_vector - neuron.weights)

                    # update positions
                    neuron.x += self.learning_rate * (
                            BMU.x - neuron.x)
                    neuron.y += self.learning_rate * (
                            BMU.y - neuron.y)
                """
                else:
                    if one_found:
                        none_found += 1
                        if none_found > 4:
                            # skip the rest of the line
                            neuron = neuron_line[-1]
                            """
        self.update_learning_rate()
        self.update_radius()
        self.iteration = self.iteration + 1

    def update_radius(self):
        # print(self._radius)
        self.radius = self.radius_0 * np.exp(-self.iteration / self.time_constant)

    def update_learning_rate(self):
        # print(self._learning_rate)
        self.learning_rate = self.learning_rate_0 * np.exp(-self.iteration / self.time_constant)

    def start(self):
        [self.update_grid(vector) for vector in self.input_data]
