# -*- coding: utf-8 -*-
import numpy as np
import random
import pandas as pd


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
        self.average_members_weights = np.array([])

    @property
    def members(self):
        return self._members

    @members.setter
    def members(self, value):
        if len(value) != 0:
            self._members = value
        else:
            raise ValueError("List of members not valid.")

    @property
    def distance_threshold(self):
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, value):
        if value <= 0:
            raise ValueError("Distance Threshold not valid.")
        else:
            self._distance_threshold = value

    def update_clustering_index(self, distance):
        # function to update an index to keep track of the "goodness" of the
        # cluster, directly proportional to the number of components and
        # inversely to the distance

        self.clustering_index = len(self.members) * (
            1 / (len(self.members) - 1) * self.clustering_index + 1 / (distance)
        )

    def average_weights(self):
        # function to return the averaged weights of the cluster

        member_weights = np.zeros(shape=len(self.members[0].weights))
        for member in self.members:
            member_weights = member_weights + member.weights
        self.average_members_weights = member_weights / len(self.members)

    def add_member(self, new_member, distance):
        # function to add a new member ot the existing cluster

        self.members = np.append(self.members, new_member)

        # update the index with the new member
        self.update_clustering_index(distance)


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


class SOM:
    def __init__(
        self,
        x_size=20,
        y_size=20,
        size_neurons=10000,
        learning_rate_0=0.5,
        radius_0=0.1,
        cluster_distance_threshold=0.04,
        input_data=None,
    ):
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
        self.time_constant = 200
        self.learning_rate_0 = learning_rate_0
        self.learning_rate = learning_rate_0
        self.radius_0 = radius_0
        self.radius = radius_0

        self.cluster_distance_threshold = cluster_distance_threshold

        self.input_data = input_data

        self.neuron_map = np.zeros(shape=(x_size, y_size), dtype=object)
        self.clusters = np.array([], dtype=object)
        self.matches_input_to_clusters = []
        self.averaged_spectra_df = []

        for i in range(self._x_size):
            for j in range(self._y_size):
                self._neuron_map[i][j] = Neuron(
                    i / x_size, j / y_size, np.random.uniform(1e-3, 9e-4, size_neurons)
                )

    @property
    def x_size(self):
        return self._x_size

    @x_size.setter
    def x_size(self, value):
        if value <= 0:
            raise ValueError("No valid x size")
        self._x_size = value

    @property
    def y_size(self):
        return self._y_size

    @y_size.setter
    def y_size(self, value):
        if value <= 0:
            raise ValueError("No valid y size")
        self._y_size = value

    @property
    def size_neurons(self):
        return self._size_neurons

    @size_neurons.setter
    def size_neurons(self, value):
        if value <= 0:
            raise ValueError("No valid neuron size")
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
        if value < 0 or value > 1:
            raise ValueError("No valid learning rate")
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
        if value < 0 or value > 1:
            raise ValueError("No valid radius")
        self._radius = value

    @property
    def cluster_distance_threshold(self):
        return self._cluster_distance_threshold

    @cluster_distance_threshold.setter
    def cluster_distance_threshold(self, value):
        if value < 0 or value > 1:
            raise ValueError("No valid distance threshold")
        self._cluster_distance_threshold = value

    @property
    def input_data(self):
        return self._input_data

    @input_data.setter
    def input_data(self, value):
        len_0 = len(value[0])
        for vector in value:
            if len(vector) != len_0:
                raise ValueError("Input data of different lengths.")
        if len(value) < 300:  # this
            raise ValueError("Too few input data.")
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

    @property
    def matches_input_to_clusters(self):
        return self._matches_input_to_clusters

    @matches_input_to_clusters.setter
    def matches_input_to_clusters(self, value):
        self._matches_input_to_clusters = value

    def find_bmu(self, input_vector):
        # compute euclidian distance from the input vector
        # to the weight vector of the neurons
        distances = np.array(
            [
                np.linalg.norm(self.neuron_map[i][j].weights - input_vector)
                for i in range(self.x_size)
                for j in range(self.y_size)
            ]
        ).reshape((self.x_size, self.y_size))

        # return the index of the neuron
        # with minimal distance (a.k.a. the best-matching unit)
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
                if (neuron.x - bmu.x) ** 2 + (
                    neuron.y - bmu.y
                ) ** 2 <= self.radius ** 2:
                    # update weights of the found neurons accordingly
                    neuron.weights = neuron.weights + self.learning_rate * (
                        input_vector - neuron.weights
                    )

                    # update positions of the found neurons accordingly
                    neuron.x += self.learning_rate * (bmu.x - neuron.x)
                    neuron.y += self.learning_rate * (bmu.y - neuron.y)
        self.update_learning_rate()
        self.update_radius()
        self.iteration = self.iteration + 1

    def update_radius(self):
        # update the radius with the known formula
        self.radius = self.radius_0 * np.exp(-self.iteration / self.time_constant)

    def update_learning_rate(self):
        # update the learning rate with the known formula
        self.learning_rate = self.learning_rate_0 * np.exp(
            -self.iteration / self.time_constant
        )

    def find_clusters(self):
        # FoF
        # make list of valid points
        list_points = [[i, j] for i in range(self.x_size) for j in range(self.y_size)]

        while list_points:
            # choose random valid point to start with
            idx = random.randint(0, len(list_points) - 1)
            start_point = list_points.pop(idx)
            start_neuron = self.neuron_map[start_point[0], start_point[1]]
            cluster = Cluster([start_neuron], self.cluster_distance_threshold)
            for point in list_points:
                # calculate distance for each point to the starting neuron
                distance = np.sqrt(
                    (self.neuron_map[point[0]][point[1]].x - start_neuron.x) ** 2
                    + (self.neuron_map[point[0]][point[1]].y - start_neuron.y) ** 2
                )
                if distance <= cluster.distance_threshold:
                    # add member to cluster
                    cluster.add_member(self.neuron_map[point[0]][point[1]], distance)
                    # remove indexes from list of valid points
                    list_points.remove(point)

            # repeat for each of the friends
            for j in range(1, len(cluster.members)):
                for point in list_points:
                    # calculate distance for each remaining point to the friends of the starting neuron
                    distance = np.sqrt(
                        (self.neuron_map[point[0]][point[1]].x - cluster.members[j].x)
                        ** 2
                        + (self.neuron_map[point[0]][point[1]].y - cluster.members[j].y)
                        ** 2
                    )
                    if distance <= cluster.distance_threshold:
                        # add member to cluster
                        cluster.add_member(
                            self.neuron_map[point[0]][point[1]], distance
                        )
                        # remove indexes from list of valid points
                        list_points.remove(point)
            # more or less subjective threshold for number of members
            if len(cluster.members) > 6:
                # calculate an average of the weights
                cluster.average_weights()
                # store the results in an array of cluster
                self.clusters = np.append(self.clusters, cluster)
            # take new random point, but do not iterate over the previously found friends

        # sort according to the higher clustering index
        self.clusters = sorted(self.clusters, key=lambda n: n.clustering_index)[::-1]
        # sort according to the lowest clustering index
        # self.clusters = sorted(self.clusters, key=lambda n: n.clustering_index)

    def match_input_to_cluster(self):
        matches_df = pd.DataFrame(columns=["Cluster_number", "Distance", "Index"])
        # associate each spectrum to a cluster, plot them
        count = 0
        for spectrum in self.input_data:
            distances = np.array([])
            for cluster in self.clusters:
                distances = np.append(
                    distances,
                    np.linalg.norm(cluster.average_members_weights - spectrum),
                )

            # store the best matching cluster with the minimal distance as an array of
            # [cluster_number, distance, index], where cluster_number is related to the ordering
            # in the clusters array, hence based on the best clustering index
            matches_df = matches_df.append(
                pd.DataFrame(
                    [
                        [
                            np.where(distances == np.amin(distances))[0][0],
                            np.amin(distances),
                            count,
                        ]
                    ],
                    columns=["Cluster_number", "Distance", "Index"],
                ),
                ignore_index=True,
            )
            count += 1

        # sort the results from lowest to highest distance for each cluster_number
        self.matches_input_to_clusters = matches_df.sort_values(
            ["Cluster_number", "Distance", "Index"], ascending=[True, True, False]
        )

    def average_spectra(self):
        # create the apposite dataframe for the averged spectra per cluster
        self.averaged_spectra_df = pd.DataFrame(
            columns=["Cluster_number", "Avg_Spectrum"]
        )
        # cycle through the clusters
        for i in range(len(self.clusters)):
            # mock spectra variable
            spectra = np.zeros(len(self.input_data[0]))

            # get spectra from i-th cluster
            df = self.matches_input_to_clusters.loc[
                self.matches_input_to_clusters["Cluster_number"] == i
            ]
            # cycle through the single spectra, average them and add them
            # to the dataframe
            for j in range(0, len(df)):
                spectra = spectra + self.input_data[df.iloc[j].Index]
            self.averaged_spectra_df = self.averaged_spectra_df.append(
                pd.DataFrame(
                    [[i, spectra / len(df)]], columns=["Cluster_number", "Avg_Spectrum"]
                ),
                ignore_index=True,
            )

    def start(self, num_cycles=1):
        # repeating the som cylce for a certain number of times,
        # with decreasing impacting parameters
        for n in range(0, num_cycles):
            self.radius = (1 / (n + 1)) * self.radius_0
            self.learning_rate = (1 / (n + 1)) * self.learning_rate_0
            [self.update_grid(vector) for vector in self.input_data]
        self.find_clusters()
        self.match_input_to_cluster()
        self.average_spectra()
