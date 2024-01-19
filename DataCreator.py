import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import os

N_IMAGES = 10 # number of images to create
N_SAMPLES = 200 # number of samples per cwt
SCALE = 15 # scale to use in cwt
FILENAME = "rod03_normal.txt"
ROD_LABEL = "falla 2"
AXIS = 'z' # axis to analyze

file_path = "./MEDICIONES/" + ROD_LABEL + "/" + FILENAME
images_directory = "./IMAGENES/" + ROD_LABEL + "/"

# check whether directory already exists
if not os.path.exists(images_directory):
    os.mkdir(images_directory)
    print("Folder %s created!" % images_directory)
else:
    print("Folder %s already exists" % images_directory)
# #WAVELET

df = pd.read_csv(file_path, sep='\t', names=['t', 'x_bits', 'y_bits', 'z_bits', 'x', 'y', 'z'])
#df['t'] = df['t'] - 0.004
del df["x_bits"]
del df["y_bits"]
del df["z_bits"]
df['t'] = df['t']/1000000
df['z'] = df['z'] - df['z'].mean()
# df_accel = df_accel.set_index('t')
# df = df_accel.reset_index()

sampling_period = (df['t'][df.shape[0] - 1] - df['t'][0]) / (df.shape[0] - 1)
sample_rate = 1 / sampling_period

random_idxs = np.random.randint(low = 0, high = df.shape[0] - 500, size = N_IMAGES)
signal = df['z'].values
widths = np.arange(1, SCALE)

for n, i in enumerate(random_idxs):
    cwtmatr, freqs = pywt.cwt(signal[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)
    # img = axs[int(i / rows), i % cols].imshow(cwtmatr, cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),
    #                                           vmin=-abs(cwtmatr).max())
    # new_cwtmatr = np.zeros((cwtmatr.shape[0] * 3, cwtmatr.shape[1]))

    # for j in range(cwtmatr.shape[0]):
    #     for k in range(cwtmatr.shape[1]):
    #         new_cwtmatr[j * 3: (j + 1) * 3, k] = cwtmatr[j, k]

    image_name = FILENAME[:-4] + "_" + str(n) + ".png"

    plt.imsave(images_directory + image_name, cwtmatr, cmap='PRGn', vmax=abs(cwtmatr).max(),
               vmin=-abs(cwtmatr).max())

plt.plot(np.arange(1, SCALE), freqs)
plt.ylim(0, 50)
plt.show()