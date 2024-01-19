import numpy as np
import pandas as pd
import os

data_directory = "./MEDICIONES/"
data_labels = os.listdir(data_directory)
label_number = list(range(len(data_labels)))

coef_means = []
coef_stds = []

for l, label in enumerate(data_labels):
    label_files_directory = os.listdir(data_directory + label)
    print("--------------")
    print("Label: ", label)

    for file in label_files_directory:
        file_path = data_directory + label + "/" + file
        df = pd.read_csv(file_path, sep='\t', names=['t', 'x_bits', 'y_bits', 'z_bits', 'x', 'y', 'z'])
        x = np.array(df['x'])
        x_bits = np.array(df['x_bits'])
        mask = x_bits != 0
        coefs = x[mask]/ x_bits[mask]
        print("File: ", file)
        mean_coef = np.mean(coefs)
        std_coef = np.std(coefs)
        coef_means.append(mean_coef)
        coef_stds.append(std_coef)
        print("  Mean: ", mean_coef)
        print("  Std: ", std_coef)


print("FINAL COEF: ", np.mean(coef_means))
print("FINAL STD: ", np.mean(coef_stds))
