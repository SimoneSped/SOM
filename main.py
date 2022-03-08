# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from synthetic_spectrum import SyntheticSpectrum
from SOM import SOM
import random


# 1. Generation and visualization of gallery of synthetic spectra
fig = plt.figure(figsize=(10, 7))
columns = 3
rows = 3
size_spectra = 200
snr = 5

for i in range(1, columns*rows+1):
    fig.add_subplot(rows, columns, i)
    num_features = np.random.randint(4) + 1
    # num_features = 2
    synth = SyntheticSpectrum(size_spectra, num_features, snr)
    plt.plot(synth.lambda_range, synth.intensities, c='black', alpha=0.5, label="N="+str(num_features)+" features")
    plt.xlim([0, size_spectra])
    plt.ylim([np.amin(synth.intensities)-10, np.amax(synth.intensities)+10])
    plt.vlines(synth.centroids, ymin=np.amin(synth.intensities)-10, ymax=np.amax(synth.intensities)+10, linestyle="--")
    plt.legend(fontsize=6)
fig.suptitle("Gallery of simulated spectra")
fig.supxlabel("Wavelength/Frequency/Kms$^{-1}$")
fig.supylabel("Intensity")
plt.tight_layout()
plt.show()


# 2. Actual SOM, with clustering and neuron map
n_data = 2000
size_spectra = 200
input_data = np.empty(
    shape=(n_data, size_spectra)
)
# generate the zoo of synthetic spectra
for i in range(0, n_data):
    num_features = np.random.randint(3) + 1
    input_data[i][:] = SyntheticSpectrum(size_spectra, num_features, snr).intensities

# initialize and start the self-organizing map
SOM = SOM(15, 15,
          size_spectra,
          0.30,
          0.20,
          0.03,
          input_data)
SOM.start(2)

# visualize the resulting clusters
fig = plt.figure(figsize=(10, 7))

best_n_to_show = 5

for cluster in SOM.clusters[:best_n_to_show]:
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
    for member in cluster.members:
        plt.scatter(member.x, member.y, s=40, c=color, edgecolors='k')
for cluster in SOM.clusters[best_n_to_show:]:
    for member in cluster.members:
        plt.scatter(member.x, member.y, s=30, c='gray')
plt.title("Resulting neuron map with " + str(len(SOM.clusters)) + " found clusters, best " + str(
    best_n_to_show) + " of them")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# visualize the best matches
x = np.arange(0, len(SOM.input_data[0]), 1)

fig = plt.figure(figsize=(10, 7))

cluster_num = 0
columns = 2
rows = 3
best_n_spectra = 10
for i in range(0, len(input_data)):
    if SOM.matches_input_to_clusters[i][1] == cluster_num:
        fig.add_subplot(rows, columns, cluster_num+1)
        for j in range(0, best_n_spectra):
            plt.plot(x, input_data[SOM.matches_input_to_clusters[i+j][0]], alpha=0.6)
        plt.xlim([0, len(x)])
        cluster_num += 1
        if cluster_num == columns*rows:
            break
fig.suptitle("Gallery of best matches of the best "+str(columns*rows)+" clusters")
fig.supxlabel("Wavelength/Frequency/Kms$^{-1}$")
fig.supylabel("Intensity")
plt.tight_layout()
plt.show()

