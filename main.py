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
    num_features = np.random.randint(4, size=1) + 1
    synth = SyntheticSpectrum(size_spectra, num_features[0], snr)
    plt.plot(synth.lambda_range, synth.intensities, c='black', alpha=0.5, label="N="+str(num_features[0])+" features")
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
n_data = 1000
size_spectra = 200
input_data = np.empty(
    shape=(n_data, size_spectra)
)
# generate the zoo of synthetic spectra
for i in range(0, n_data):
    num_features = np.random.randint(3, size=1) + 1
    input_data[i][:] = SyntheticSpectrum(size_spectra, num_features[0], snr).intensities

# initialize and start the self-organizing map
SOM = SOM(20, 20,
          size_spectra,
          0.30,
          0.20,
          0.03,
          input_data)
SOM.start()

# visualize the results
fig = plt.figure(figsize=(10, 7))

best_n_to_show = 10

for cluster in SOM.clusters[:best_n_to_show]:
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
    for member in cluster.members:
        plt.scatter(member.x, member.y, s=40, c=color, edgecolors='k')
for cluster in SOM.clusters[best_n_to_show:]:
    for member in cluster.members:
        plt.scatter(member.x, member.y, s=30, c='gray')
plt.title("Resulting neuron map with "+str(len(SOM.clusters))+" found clusters, best "+str(best_n_to_show)+" of them")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 3. Associate each spectrum to a cluster, plot them
cats = []
for spectrum in input_data:
    distances = np.array([])
    for cluster in SOM.clusters[:5]:
        distances_members = np.array([])
        for member in cluster.members:
            distances_members = np.append(distances_members, np.linalg.norm(member.weights - spectrum))
        distances = np.append(distances, np.mean(distances_members))
    cats.append([len(cats), np.where(distances == np.amin(distances))[0][0], np.amin(distances)])

cats_0 = []
cats_1 = []
cats_2 = []
cats_3 = []
for i in range(0, len(cats)):
    if cats[i][1] == 0:
        cats_0.append(cats[i])
    else:
        if cats[i][1] == 1:
            cats_1.append(cats[i])
        else:
            if cats[i][1] == 2:
                cats_2.append(cats[i])
            else:
                if cats[i][1] == 3:
                    cats_3.append(cats[i])

cats_0 = sorted(cats_0, key=lambda n: n[2])
cats_1 = sorted(cats_1, key=lambda n: n[2])
cats_2 = sorted(cats_2, key=lambda n: n[2])
cats_3 = sorted(cats_3, key=lambda n: n[2])

# visualize the results
fig, axs = plt.subplots(2, 2, figsize=(10, 7))
x = np.arange(0, 200, 1)

for i in range(0, 5):
    axs[0][0].plot(x, input_data[cats_0[i][0]], alpha=0.5)
axs[0][0].set_xlim([0, 200])
axs[0][0].set_title("Best matches for the first best cluster")
axs[0][0].set_xticks([])
for i in range(0, 5):
    axs[0][1].plot(x, input_data[cats_1[i][0]], alpha=0.5)
axs[0][1].set_xlim([0, 200])
axs[0][1].set_title("Best matches for the second best cluster")
axs[0][1].set_xticks([])
for i in range(0, 5):
    axs[1][0].plot(x, input_data[cats_2[i][0]], alpha=0.5)
axs[1][0].set_xlim([0, 200])
axs[1][0].set_title("Best matches for the third best cluster")
for i in range(0, 5):
    axs[1][1].plot(x, input_data[cats_3[i][0]], alpha=0.5)
axs[1][1].set_xlim([0, 200])
axs[1][1].set_title("Best matches for the fourth best cluster")

fig.supxlabel("Wavelength/Frequency/Kms$^{-1}$")
fig.supylabel("Intensity")
plt.show()
