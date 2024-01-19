import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import os

SCALE = 31 # scale to use in cwt
FILENAME = "rod03_golpes.txt"
AXIS = 'z' # axis to analyze

file_path = "./MEDICIONES/falla 3/" + FILENAME
# images_directory = "./IMAGENES/" + FILENAME[:-4] + "/"

df = pd.read_csv(file_path, sep='\t', names=['t', 'x_bits', 'y_bits', 'z_bits', 'x', 'y', 'z'])
#df['t'] = df['t'] - 0.004
del df["x_bits"]
del df["y_bits"]
del df["z_bits"]
df['t'] = df['t']/1000000

df['t'] = df['t']/1000000
print(df['z'].mean())
df['z'] = df['z'] - df['z'].mean()
print(df['z'].mean())

plt.plot(df['t'], df['z'])
plt.show()