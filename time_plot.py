import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

import pywt
import os

DATA_DIRECTORY = "/home/ronaldocd/Desktop/RC/Wavelet/TESIS/MEDICIONES/"

rod = "rod02_normal.txt"
# fallas = ["falla_0/", "falla_1/", "falla_2/", "falla_3/"]
fallas = ["falla_0/"]
CALIB_FACTOR = 0.0025527271511137624

fig, axs = plt.subplots(4, 3)
fig2, axs2 = plt.subplots(4, 3)

# Note the extra 'r' at the front

for (i, f) in enumerate(fallas):
    path = DATA_DIRECTORY + f + rod
    df = pd.read_csv(path, sep='\t', names=['t', 'x_bits', 'y_bits', 'z_bits', 'x', 'y', 'z'])
    df['t'] = df['t'] / 1000000
    df['t'] = df['t'] - df['t'][0]
    sampling_period = (df['t'][df.shape[0] - 1] - df['t'][0]) / (df.shape[0] - 1)
    signal_x = df['x_bits'] * CALIB_FACTOR / 9.81
    signal_y = df['y_bits'] * CALIB_FACTOR / 9.81
    signal_z = df['z_bits'] * CALIB_FACTOR / 9.81

    signal_x = (signal_x - np.mean(signal_x))
    signal_y = (signal_y - np.mean(signal_y))
    signal_z = (signal_z - np.mean(signal_z))

    axs[i][0].plot(df['t'], signal_x, label='X', color='red', alpha=1, linewidth=0.3)
    axs[i][1].plot(df['t'], signal_y, label='X', color='green', alpha=1, linewidth=0.3)
    axs[i][2].plot(df['t'], signal_z, label='X', color='blue', alpha=1, linewidth=0.3)

    axs[i][0].set_xlim([0, 30])
    axs[i][0].set_ylim([-2, 2])

    axs[i][1].set_xlim([0, 30])
    axs[i][1].set_ylim([-2, 2])

    axs[i][2].set_xlim([0, 30])
    axs[i][2].set_ylim([-2, 2])

    fft_x = rfft(np.array(signal_x))
    fft_y = rfft(np.array(signal_y))
    fft_z = rfft(np.array(signal_z))
    xf = rfftfreq(len(signal_x), sampling_period)
    axs2[i][0].plot(xf, np.abs(fft_x), label='X', color='red', alpha=1, linewidth=0.3)
    axs2[i][1].plot(xf, np.abs(fft_y), label='X', color='green', alpha=1, linewidth=0.3)
    axs2[i][2].plot(xf, np.abs(fft_z), label='X', color='blue', alpha=1, linewidth=0.3)

    axs2[i][0].set_xlim([0, 500])
    axs2[i][0].set_ylim([0, 2000])

    axs2[i][1].set_xlim([0, 500])
    axs2[i][1].set_ylim([0, 2000])

    axs2[i][2].set_xlim([0, 500])
    axs2[i][2].set_ylim([0, 2000])

for ax in axs.flat:
    ax.set(xlabel='Tiempo [s]', ylabel="g")

for ax in fig.get_axes():
    ax.label_outer()

for ax in axs2.flat:
    ax.set(xlabel='Frecuencia [Hz]', ylabel="g")

for ax in fig2.get_axes():
    ax.label_outer()

plt.show()
plt.legend(["X", "Y", "Z"])