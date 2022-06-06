# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import timeit

from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits

import pandas as pd
import warnings

from synthetic_spectrum import SyntheticSpectrum
from SOM import SOM


def generate_example_spectra(
    probabilities, size_spectra, max_components, snr, separation, desired_features=None
):
    # function to generate and show typical mock spectra

    fig = plt.figure(figsize=(10, 7))
    plt.style.use("bmh")

    # define how many columns and rows, and the probabilities
    columns = 3
    rows = 3

    values = np.array([i for i in range(max_components + 1)])
    probabilities = [0.3, 0.25, 0.2, 0.15, 0.10]

    spectra = np.array([], dtype=object)

    # generate the spectra
    for i in range(0, columns * rows):
        num_features = np.random.choice(values, p=probabilities)
        spectra = np.append(
            spectra,
            SyntheticSpectrum(
                size_spectra, num_features, snr, separation, desired_features
            ),
        )

    # normalization
    list_max = []
    list_min = []
    for i in range(len(spectra)):
        list_max.append(np.amax(spectra[i].intensities))
        list_min.append(np.amin(spectra[i].intensities))

    for i in range((len(spectra))):
        spectra[i].intensities = (spectra[i].intensities - np.amin(list_min)) / (
            np.amax(list_max) - np.amin(list_min)
        )

    # plotting
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)

        plt.plot(
            spectra[i - 1].lambda_range,
            spectra[i - 1].intensities,
            c="black",
            alpha=0.5,
            label="N=" + str(spectra[i - 1].num_features) + " features",
        )
        plt.xlim([0, size_spectra])
        plt.ylim([-0.5, 1])

        plt.vlines(
            spectra[i - 1].centroids,
            ymin=np.amin(spectra[i - 1].intensities) - 10,
            ymax=np.amax(spectra[i - 1].intensities) + 10,
            linestyle="--",
        )
        plt.legend(fontsize=6)

    fig.suptitle("Gallery of simulated spectra")
    fig.supxlabel("Channels")
    fig.supylabel("Intensity")
    plt.tight_layout()
    plt.show()


def input_handler(path, slice_size=100):
    # function to turn a given data-cube into single spectra of same dimension
    # dealing with the presence of nan values

    # get the data from the given path
    image_data = fits.getdata(path, ext=0)

    # discard the first dimension
    data_cube = image_data[0, :, :, :]  # vel_rad, RA, Dec, (300x871x973)

    input_data = []
    input_spectra = []

    # dissect each slice of the 3D cube into single lines (e.g. columns or rows)
    """
    for i in range(0, data_cube.shape[0]):
        for j in range(0, data_cube.shape[1]):
            input_data.append(data_cube[i, j])
            # input_data[(i*data_cube.shape[0])+j][:] = data_cube[i, j]
    """

    # alternatively
    for i in range(0, data_cube.shape[0]):
        for j in range(0, data_cube.shape[2]):
            input_data.append(data_cube[i, :, j])

    # slicing each line in single spectra of a given length
    for i in range(0, len(input_data)):
        count = 0
        for j in range(0, len(input_data[i])):
            if input_data[i][j] == input_data[i][j]:
                # is not a nan
                count += 1
                if count == slice_size:
                    # enough not-nan in a row, save spectra
                    input_spectra.append(input_data[i][j - count + 1 : j + 1])
                    # reset counter and move ahead in the line
                    count = 0
            else:
                # is a nan, reset counter
                count = 0
    return input_spectra


def visualize_neurons_map(SOM, best_n_to_highlight=5):
    # function to visualize the current configuration of the
    # neural map, hihglighting some of the clusters

    fig = plt.figure(figsize=(10, 7))

    plt.style.use("bmh")

    # plot with colour the best n
    for cluster in SOM.clusters[:best_n_to_highlight]:
        # generate random colour for the cluster
        color = ["#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])]

        for member in cluster.members:
            plt.scatter(member.x, member.y, s=40, c=color, edgecolors="k")

    # plot in grey the remaining
    for cluster in SOM.clusters[best_n_to_highlight:]:
        for member in cluster.members:
            plt.scatter(member.x, member.y, s=30, c="gray")
    plt.title(
        "Resulting neuron map with "
        + str(len(SOM.clusters))
        + " found clusters, best "
        + str(best_n_to_highlight)
        + " of them"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def visualize_best_n_clusters(SOM, number_of_clusters, number_of_spectra):
    # overview of n cluster, with a number of characteristic spectra
    # to show the defining feature of the cluster

    x = np.arange(0, len(SOM.input_data[0]), 1)
    fig = plt.figure(figsize=(10, 7))
    plt.style.use("bmh")

    # define how many to plot
    columns = 2
    rows = 4

    # check for validity
    if number_of_clusters > len(SOM.clusters):
        number_of_clusters = len(SOM.clusters)

    # plot
    for i in range(0, number_of_clusters):
        fig.add_subplot(rows, columns, i + 1)
        df = SOM.matches_input_to_clusters.loc[
            SOM.matches_input_to_clusters["Cluster_number"] == i
        ]
        for j in range(0, number_of_spectra):
            plt.plot(x, input_data[df.iloc[j].Index], c="black", alpha=0.5)
            plt.xlim([0, len(x)])

    fig.suptitle(
        "Best matches for the best "
        + str(number_of_clusters)
        + " clusters (out of a total of "
        + str(len(SOM.clusters))
        + " clusters)"
    )
    fig.supxlabel("Channels")
    fig.supylabel("Intensity")
    plt.tight_layout()
    plt.show()


def visualize_cluster(SOM, cluster_number, num_pages):
    # function to better show the properties of a single cluster, done
    # by plottin on a 4x4 grid the best 16 spectra, and the
    # corresponding averaged spectra as well. This can be
    # asked a number of times, expressed throught the number of pages.

    x = np.arange(0, len(SOM.input_data[0]), 1)
    plt.style.use("bmh")

    # define dimension of plot grid
    columns = 4
    rows = 4

    # get data from the dataframes for the given cluster
    df = SOM.matches_input_to_clusters.loc[
        SOM.matches_input_to_clusters["Cluster_number"] == cluster_number
    ]
    df_averaged_spectra = SOM.averaged_spectra_df.loc[
        SOM.averaged_spectra_df.Cluster_number == cluster_number
    ]

    # orderly divides the spectra on the required pages to show
    # without running into exceptions of matplotlib
    if len(df) >= num_pages * (columns * rows):
        for i in range(0, num_pages):
            fig = plt.figure(figsize=(10, 7))

            for j in range(0, columns * rows):
                fig.add_subplot(rows, columns, j + 1)
                plt.plot(
                    x,
                    input_data[df.iloc[i * (columns * rows) + j].Index],
                    c="black",
                    alpha=0.5,
                )
                plt.plot(
                    x,
                    df_averaged_spectra["Avg_Spectrum"][cluster_number],
                    c="red",
                    alpha=0.5,
                )
                plt.xlim([0, len(x)])
            fig.suptitle(
                "Representative matches and averaged spectrum of cluster no."
                + str(cluster_number + 1)
                + " (out of a total of "
                + str(len(df))
                + " spectra)"
            )
            fig.supxlabel("Channels")
            fig.supylabel("Intensity")
            plt.tight_layout()
            plt.show()
    else:
        rest = len(df) // (columns * rows)

        for i in range(0, rest):
            fig = plt.figure(figsize=(10, 7))

            for j in range(0, columns * rows):
                fig.add_subplot(rows, columns, j + 1)
                plt.plot(
                    x,
                    input_data[df.iloc[i * (columns * rows) + j].Index],
                    c="black",
                    alpha=0.5,
                )
                plt.plot(
                    x,
                    df_averaged_spectra["Avg_Spectrum"][cluster_number],
                    c="red",
                    alpha=0.5,
                )
                plt.xlim([0, len(x)])
            fig.suptitle(
                "Representative matches and averaged spectrum of cluster no."
                + str(cluster_number + 1)
                + " (out of a total of "
                + str(len(df))
                + " spectra)"
            )
            fig.supxlabel("Channels")
            fig.supylabel("Intensity")
            plt.tight_layout()
            plt.show()

        fig = plt.figure(figsize=(10, 7))

        for k in range(0, len(df) - rest * (columns * rows)):
            fig.add_subplot(rows, columns, k + 1)
            plt.plot(
                x,
                input_data[df.iloc[rest * (columns * rows) + k].Index],
                c="black",
                alpha=0.5,
            )
            plt.plot(
                x,
                df_averaged_spectra["Avg_Spectrum"][cluster_number],
                c="red",
                alpha=0.5,
            )
            plt.xlim([0, len(x)])
        fig.suptitle(
            "Representative matches and averaged spectrum of cluster no."
            + str(cluster_number + 1)
            + " (out of a total of "
            + str(len(df))
            + " spectra)"
        )
        fig.supxlabel("Channels")
        fig.supylabel("Intensity")
        plt.tight_layout()
        plt.show()


def calculate_data_model_deviation(data, model):
    # calculate the model to data deviation in form of the chi2 metric
    return np.sum((model - data) ** 2)


def fit_clusters(SOM, df_fit, input_data):
    # function to prompt the user for the required guesses for the
    # fits of the single clusters

    # get how many clusters are to be fitted
    to_fit_clusters = len(SOM.clusters)

    for cluster_number in range(0, to_fit_clusters):
        # first show the corresponding cluster,
        # to enable to user to make a guess
        visualize_cluster(SOM, cluster_number, 1)

        # get guess for the number of components recognizable
        number_guessed_components = int(
            input(
                "-- Cluster "
                + str(cluster_number + 1)
                + " -- Insert the guess for the number of components: "
            )
        )
        while number_guessed_components < 0 or number_guessed_components > 5:
            print("- Invalid guess.")
            number_guessed_components = int(
                input(
                    "-- Cluster "
                    + str(cluster_number + 1)
                    + " -- Insert the guess for the number of components: "
                )
            )

        # get guess for the single defining parameters of each
        guesses_centroids = np.array([])
        guesses_amplitudes = np.array([])
        guesses_stddevs = np.array([])
        if number_guessed_components > 0:

            for i in range(0, number_guessed_components):
                print("")
                print("===============================================")
                print(
                    "-- Guesses for the component no. "
                    + str(i + 1)
                    + "/"
                    + str(number_guessed_components)
                    + " --"
                )
                print("===============================================")

                # input and validty check of the centroid
                guess_centroid = int(input("- Insert the guess for the centroid: "))
                while guess_centroid < 0 or guess_centroid >= len(input_data[0]):
                    print("- Invalid guess.")
                    guess_centroid = int(input("- Insert the guess for the centroid: "))

                guesses_centroids = np.append(guesses_centroids, guess_centroid)

                # input and validty check of the amplitude
                guess_amplitude = float(input("- Insert the guess for the amplitude: "))
                while guess_amplitude <= 0 or guess_amplitude > 1.5:
                    print("- Invalid guess.")
                    guess_amplitude = float(
                        input("- Insert the guess for the amplitude: ")
                    )

                guesses_amplitudes = np.append(guesses_amplitudes, guess_amplitude)

                # input and validty check of the standard deviation
                guess_stddev = float(input("- Insert the guess for the std.: "))
                while guess_stddev <= 0 or guess_stddev >= len(input_data[0]):
                    print("- Invalid guess.")
                    guess_stddev = float(input("- Insert the guess for the std.: "))

                guesses_stddevs = np.append(guesses_stddevs, guess_stddev)

            # astropy.modeling fitting
            df = SOM.matches_input_to_clusters.loc[
                SOM.matches_input_to_clusters["Cluster_number"] == cluster_number
            ]

            x = np.arange(0, len(SOM.input_data[0]), 1)

            gg_init = np.array([], dtype=object)
            for i in range(0, number_guessed_components):
                gg_init = np.append(
                    gg_init,
                    models.Gaussian1D(
                        amplitude=guesses_amplitudes[i],
                        mean=guesses_centroids[i],
                        stddev=guesses_stddevs[i],
                    ),
                )

            fitter = fitting.LevMarLSQFitter()

            for i in range(0, len(df)):

                data = input_data[df.iloc[i].Index]
                with warnings.catch_warnings():
                    # Ignore a warning on clipping to bounds from the fitter
                    warnings.simplefilter("ignore", AstropyWarning)
                    gg_fit = fitter(np.sum(gg_init), x, data)

                df_fit = df_fit.append(
                    pd.DataFrame(
                        [
                            [
                                cluster_number,
                                gg_fit,
                                df.iloc[i].Index,
                                calculate_data_model_deviation(data, gg_fit(x)),
                            ]
                        ],
                        columns=["Cluster_number", "fitter", "Index_spectrum", "Chi2"],
                    )
                )

            # order spectra according to chi2
            df_fit = df_fit.sort_values(
                ["Cluster_number", "Chi2"], ascending=[True, True]
            )

    return df_fit


def visualize_fits(SOM, df_fit, cluster_number, num_pages):
    # function to visualize spectra and fits. It works similarly to the
    # visualize_cluster function, with addition of the fit on top of it and
    # without the averaged spectra

    df_fit = df_fit.loc[df_fit.Cluster_number == cluster_number]
    plt.style.use("bmh")

    if len(df_fit) == 0:
        return 0

    x = np.arange(0, len(SOM.input_data[0]), 1)

    columns = 4
    rows = 4

    if len(df_fit) >= num_pages * (columns * rows):
        for i in range(0, num_pages):
            fig = plt.figure(figsize=(10, 7))

            for j in range(0, columns * rows):
                fig.add_subplot(rows, columns, j + 1)
                plt.plot(
                    x,
                    input_data[df_fit.iloc[i * (columns * rows) + j].Index_spectrum],
                    c="black",
                    alpha=0.5,
                )
                plt.plot(
                    x,
                    df_fit.iloc[i * (columns * rows) + j].fitter(x),
                    c="orange",
                    alpha=0.6,
                )
                plt.xlim([0, len(x)])
            fig.suptitle(
                "Fits for the cluster no."
                + str(cluster_number)
                + " (out of a total of "
                + str(len(df_fit))
                + " spectra)"
            )
            fig.supxlabel("Channels")
            fig.supylabel("Intensity")
            plt.tight_layout()
            plt.show()
    else:
        rest = len(df_fit) // (columns * rows)

        for i in range(0, rest):
            fig = plt.figure(figsize=(10, 7))

            for j in range(0, columns * rows):
                fig.add_subplot(rows, columns, j + 1)
                plt.plot(
                    x,
                    input_data[df_fit.iloc[i * (columns * rows) + j].Index],
                    c="black",
                    alpha=0.5,
                )
                plt.plot(
                    x,
                    df_fit.iloc[i * (columns * rows) + j].fitter(x),
                    c="orange",
                    alpha=0.6,
                )
                plt.xlim([0, len(x)])
            fig.suptitle(
                "Fits for the cluster no."
                + str(cluster_number)
                + " (out of a total of "
                + str(len(df_fit))
                + " spectra)"
            )
            fig.supxlabel("Channels")
            fig.supylabel("Intensity")
            plt.tight_layout()
            plt.show()

        fig = plt.figure(figsize=(10, 7))

        for k in range(0, len(df_fit) - rest * (columns * rows)):
            fig.add_subplot(rows, columns, k + 1)
            plt.plot(
                x,
                input_data[df_fit.iloc[rest * (columns * rows) + k].Index],
                c="black",
                alpha=0.5,
            )
            plt.plot(
                x,
                df_fit.iloc[i * (columns * rows) + j].fitter(x),
                c="orange",
                alpha=0.6,
            )
            plt.xlim([0, len(x)])
        fig.suptitle(
            "Fits for the cluster no."
            + str(cluster_number)
            + " (out of a total of "
            + str(len(df_fit))
            + " spectra)"
        )
        fig.supxlabel("Channels")
        fig.supylabel("Intensity")
        plt.tight_layout()
        plt.show()


def visualize_chi2(SOM, df_fit):
    # function to visualize the normalized distribution of the chi2 values
    # derived from the deviation model to data after the fit process

    fig = plt.figure(figsize=(10, 7))
    plt.style.use("bmh")

    # Normalize and  calculate percentiles
    max_chi2 = np.amax(df_fit["Chi2"])
    min_chi2 = np.amin(df_fit["Chi2"])
    df_fit["Chi2"] = (df_fit["Chi2"] - min_chi2) / (max_chi2 - min_chi2)

    quant_95, quant_75, quant_50 = (
        np.nanpercentile(df_fit["Chi2"], 95),
        np.nanpercentile(df_fit["Chi2"], 75),
        np.nanpercentile(df_fit["Chi2"], 50),
    )
    plt.hist(df_fit["Chi2"], bins=75, alpha=0.65)

    # [quantile, opacity, length]
    quants = [[quant_50, 1, 0.36], [quant_75, 0.8, 0.46], [quant_95, 0.6, 0.56]]

    # Plot with adequate labels
    for i in quants:
        plt.axvline(i[0], alpha=i[1], ymax=i[2], linestyle=":")
    plt.text(quant_50, quants[0][2] + 70, "50th", size=12, alpha=0.8)
    plt.text(quant_75, quants[1][2] + 120, "75th", size=12, alpha=0.8)
    plt.text(quant_95, quants[2][2] + 170, "95th Percentile", size=12, alpha=0.8)

    plt.tick_params(left=False, bottom=False)
    plt.ylabel("# Fits")
    plt.xlabel("Chi2 values")
    plt.title(
        "Distribution of chi2 metric for the "
        + str(len(df_fit["Chi2"]))
        + " fitted spectra"
    )


# 0 - Example of spectra
"""
probabilities = [0.3, 0.25, 0.2, 0.15, 0.10]
size_spectra = 200
snr = 5
max_components = 4
separation = 10
desired_features = [[100, 250, 5], [150, 265, 6]]
generate_example_spectra(
    probabilities,
    size_spectra,
    max_components,
    snr,
    separation,
    desired_features
    )

"""
# 1 - SOM

"""
# 1.1 - generate the zoo of synthetic spectra
number_spectra = 300
size_spectra = 200
max_components = 4
separation = 5
probabilities = [0.3, 0.25, 0.2, 0.15, 0.10]
# probabilities = [0.1, 0.2, 0.4, 0.2, 0.10]
snr = 5

input_data = np.zeros(
    shape=(number_spectra, size_spectra)
)
values = np.array([i for i in range(max_components+1)])

for i in range(0, number_spectra):
    num_features = np.random.choice(values, p=probabilities)
    input_data[i][:] = SyntheticSpectrum(
        size_spectra, num_features, snr, separation).intensities
"""

# 1.1/bis -  read the data from file
size_spectra = 125
input_data = input_handler("B213_C18O10.fits", size_spectra)
# print(len(input_data))

"""
# 1.1/tris - generate the zoo of synthetic spectra with desired feautures
number_spectra = 50000
size_spectra = 100
max_components = 4
separation = 5
probabilities = [0.3, 0.25, 0.2, 0.15, 0.10]
# probabilities = [0.1, 0.2, 0.4, 0.2, 0.10]
snr = 5
num_desired_spectra = [2000, 2000]

input_data = np.zeros(
    shape=(np.sum(num_desired_spectra), size_spectra)
)
values = np.array([i for i in range(max_components+1)])

desired_features = [
    [[50, 280, 3], [55, 280, 3]],
    [[53, 280, 4]]
    ]

for i in range(0, len(desired_features)):
    for j in range(int(np.sum(num_desired_spectra[:i])), num_desired_spectra[i]+int(np.sum(num_desired_spectra[:i]))):
        num_features = np.random.choice(values, p=probabilities)
        input_data[j][:] = SyntheticSpectrum(
            size_spectra,
            num_features,
            snr,
            separation,
            desired_features[i]
        ).intensities
"""

np.random.shuffle(input_data)

# 1.2 - normalization: TODO
max_intensity = np.amax(input_data)

# min_intensity = np.amin(input_data)
# input_data = [((input_data[i][:] - min_intensity) / (max_intensity - min_intensity))
# for i in range(0, len(input_data))]

input_data = [(input_data[i][:]) / (max_intensity) for i in range(0, len(input_data))]

# 1.3 - initialize and start the self-organizing map
# starttime = timeit.default_timer()
som = SOM(25, 25, size_spectra, 0.2, 0.2, 0.02, input_data)
som.start()

# print("The time difference is :", timeit.default_timer() - starttime)

# 2 - Visualize the results

# 2.1 - visualize the distribution of the neurons
visualize_neurons_map(som)

# 2.2 - visualize the best n clusters
visualize_best_n_clusters(som, 8, 5)

# 3 - Fitting
df_fit = pd.DataFrame(columns=["Cluster_number", "fitter", "Index_spectrum", "Chi2"])

df_fit = fit_clusters(som, df_fit, input_data)

# 4 - Visualize the results
visualize_chi2(som, df_fit)

# 5 - visualize and organize the results
for i in range(0, len(som.clusters)):
    visualize_fits(som, df_fit, i, 1)

# Extra - Rerun the SOM for the worse fits?
