# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from synthetic_feature import SyntheticSpectrum
from SOM import SOM

"""
fig = plt.figure(figsize=(10, 7))
columns = 3
rows = 3

for i in range(1, columns*rows+1):
    fig.add_subplot(rows, columns, i)
    num_features = np.random.randint(9, size=1) + 1
    synth = SyntheticSpectrum(500, num_features[0], 0.5)
    plt.plot(synth.lambda_range, synth.intensities, c='orange', alpha=0.5)
    plt.xlim([0, 500])
fig.suptitle("Simulated spectra")
fig.supxlabel("Wavelength/Frequency")
fig.supylabel("Intensity")
plt.tight_layout()
plt.show()
"""

n_data = 3000
size_spectra = 10000
step = 1
input_data = np.empty(
    shape=(n_data, size_spectra)
)
for i in range(0, n_data):
    num_features = np.random.randint(1, size=4) + 1
    input_data[i][:] = SyntheticSpectrum(10000, num_features[0]).intensities
SOM = SOM(20, 20, size_spectra, 0.4, 0.4, input_data)
SOM.start()

fig = plt.figure(figsize=(8, 5))

for i in range(SOM.x_size):
    for j in range(SOM.y_size):
        plt.scatter(SOM.neuron_map[i][j].x, SOM.neuron_map[i][j].y, alpha=0.3)
plt.show()

