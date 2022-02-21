# -*- coding: utf-8 -*-


class Neuron:

    def __init__(self, x_0, y_0, weights):
        """
        Class which creates the single neurons of the SOM grid
        :type x_0: float
        :type y_0: float
        :type weights: numpy array
        """
        self.x = x_0
        self.y = y_0
        self.weights = weights

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value
