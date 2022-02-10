# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from synthetic_feature import SyntheticSpectrum

fig = plt.figure(figsize=(10, 7))
columns = 5
rows = 5

for i in range(1, columns*rows+1):
    fig.add_subplot(rows, columns, i)
    num_features = np.random.randint(19, size=1) + 1
    synth = SyntheticSpectrum([0, 10000], 0.1, num_features[0])
    plt.plot(synth.lambda_range, synth.intensities, c='orange', alpha=0.7)
    plt.xlim([synth.interval[0], synth.interval[1]])
fig.suptitle("Simulated spectra")
fig.supxlabel("Wavelength/Frequency")
fig.supylabel("Intensity")
plt.tight_layout()
plt.show()
