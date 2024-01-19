import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import os

IMAGES_DIRECTORY = "/home/ronaldocd/Desktop/RC/Wavelet/TESIS/IMAGENES/"
N_IMAGES_TRAINING_SET = 100#100 # number of images to create
N_IMAGES_TEST_SET = 50#50
N_IMAGES = N_IMAGES_TRAINING_SET + N_IMAGES_TEST_SET
N_SAMPLES = 200 # number of samples per cwt
SCALE = 31 # scale to use in cwt
CALIB_FACTOR = 0.0025527271511137624

AXIS = 'z_bits' # axis to analyze

def create_images(file_path, label, file):
    df_train_summary = pd.DataFrame(columns=['image_path', 'label', 'label_idx'])
    df_test_summary = pd.DataFrame(columns=['image_path', 'label', 'label_idx'])
    df = pd.read_csv(file_path, sep='\t', names=['t', 'x_bits', 'y_bits', 'z_bits', 'x', 'y', 'z'])
    # del df["x_bits"]
    # del df["y_bits"]
    # del df["z_bits"]
    df['t'] = df['t'] / 1000000
    sampling_period = (df['t'][df.shape[0] - 1] - df['t'][0]) / (df.shape[0] - 1)
    sampling_period = np.mean(np.array(df['t'][1:50000]) - np.array(df['t'][0:50000-1]))
    # df = df.iloc[:, [1,2,3]]
    #
    # df = pd.DataFrame(df_scaled, columns=[
    #     'x', 'y', 'z'])

    # print("   ")
    print("Label: ", label)
    print("File: ", file)
    print("Sampling frequency: ", 1 / sampling_period)
    # print(df.describe())

    random_idxs = np.random.randint(low=0, high=df.shape[0] - 500, size=N_IMAGES)
    random_idxs_train = random_idxs[:N_IMAGES_TRAINING_SET]
    random_idxs_test = random_idxs[N_IMAGES_TRAINING_SET:]
    # signal = df[AXIS].values
    signal_x = df['x_bits'] * CALIB_FACTOR
    signal_y = df['y_bits'] * CALIB_FACTOR
    signal_z = df['z_bits'] * CALIB_FACTOR

    signal_x = (signal_x - np.mean(signal_x)) / np.std(signal_x)
    signal_y = (signal_y - np.mean(signal_y)) / np.std(signal_y)
    signal_z = (signal_z - np.mean(signal_z)) / np.std(signal_z)
    # print("Signal mean: ", np.mean(signal))
    # print("Signal std: ", np.std(signal))

    widths = np.arange(1, SCALE)
    cwtmatr = np.zeros((SCALE - 1, N_SAMPLES, 3))

    for n, i in enumerate(random_idxs_train):
        cwtmatr_x, freqs = pywt.cwt(signal_x[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)
        cwtmatr_y, freqs = pywt.cwt(signal_y[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)
        cwtmatr_z, freqs = pywt.cwt(signal_z[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)

        cwtmatr_x = (cwtmatr_x - np.min(cwtmatr_x)) / (np.max(cwtmatr_x) - np.min(cwtmatr_x))
        cwtmatr_y = (cwtmatr_y - np.min(cwtmatr_y)) / (np.max(cwtmatr_y) - np.min(cwtmatr_y))
        cwtmatr_z = (cwtmatr_z - np.min(cwtmatr_z)) / (np.max(cwtmatr_z) - np.min(cwtmatr_z))

        cwtmatr[:, :, 0] = cwtmatr_x
        cwtmatr[:, :, 1] = cwtmatr_y
        cwtmatr[:, :, 2] = cwtmatr_z

        # print("CWT X mean: ", np.mean(cwtmatr_x))
        # print("CWT X std: ", np.std(cwtmatr_x))
        # print("CWT X max: ", np.max(cwtmatr_x))
        # print("CWT X max: ", np.min(cwtmatr_x))
        # print("CWT X sum: ", np.sum(cwtmatr_x > 1))
        #
        # print("CWT Y mean: ", np.mean(cwtmatr_y))
        # print("CWT Y std: ", np.std(cwtmatr_y))
        # print("CWT Y sum: ", np.sum(cwtmatr_y > 1))
        #
        # print("CWT Z mean: ", np.mean(cwtmatr_z))
        # print("CWT Z std: ", np.std(cwtmatr_z))
        # print("CWT Z sum: ", np.sum(cwtmatr_z > 1))
        image_name = label + "_" + file[:-4] + "_" + str(n) + ".png"
        # image_name_2 = label + "_" + file[:-4] + "_" + str(n) + "pil.png"

        image_path = IMAGES_DIRECTORY + "train" + "/" + label + "/" + image_name
        # image_path = IMAGES_DIRECTORY + "train" + "/" + label + "_" + AXIS + "/" + image_name

        # plt.imsave(image_path, cwtmatr, cmap='PRGn', vmax=abs(cwtmatr).max(),
        #            vmin=-abs(cwtmatr).max())

        plt.imsave(image_path, cwtmatr, cmap='PRGn', vmax=1, vmin=0)

        # plt.imsave(image_path, cwtmatr, cmap='PRGn', vmax=np.std(cwtmatr), vmin=-np.std(cwtmatr))

        new_row_train = {'image_path': image_path, 'label':label, 'label_idx':label[6]}
        df_train_summary.loc[len(df_train_summary)] = new_row_train
        # Use the loc method to add the new row to the DataFrame
        # print(freqs)

    for n, i in enumerate(random_idxs_test):
        cwtmatr_x, freqs = pywt.cwt(signal_x[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)
        cwtmatr_y, freqs = pywt.cwt(signal_y[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)
        cwtmatr_z, freqs = pywt.cwt(signal_z[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)

        cwtmatr_x = (cwtmatr_x - np.min(cwtmatr_x)) / (np.max(cwtmatr_x) - np.min(cwtmatr_x))
        cwtmatr_y = (cwtmatr_y - np.min(cwtmatr_y)) / (np.max(cwtmatr_y) - np.min(cwtmatr_y))
        cwtmatr_z = (cwtmatr_z - np.min(cwtmatr_z)) / (np.max(cwtmatr_z) - np.min(cwtmatr_z))

        cwtmatr[:, :, 0] = cwtmatr_x
        cwtmatr[:, :, 1] = cwtmatr_y
        cwtmatr[:, :, 2] = cwtmatr_z

        image_name = label + "_" + file[:-4] + "_" + str(n) + ".png"
        image_path = IMAGES_DIRECTORY + "test" + "/" + label + "/" + image_name
        # image_path = IMAGES_DIRECTORY + "test" + "/" + label + "_" + AXIS + "/" + image_name

        plt.imsave(image_path, cwtmatr, cmap='PRGn', vmax=1, vmin=0)

        # plt.imsave(image_path, cwtmatr, cmap='gray', vmax=abs(cwtmatr).max(),
        #            vmin=-abs(cwtmatr).max())
        # plt.imsave(image_path, cwtmatr, cmap='PRGn', vmax=np.std(cwtmatr), vmin=-np.std(cwtmatr))
        new_row_test = {'image_path': image_path, 'label': label, 'label_idx': label[6]}
        df_test_summary.loc[len(df_test_summary)] = new_row_test

    return df_train_summary, df_test_summary


data_directory = "./MEDICIONES/"
data_labels = os.listdir(data_directory)
label_number = list(range(len(data_labels)))

train_df_csv = []
test_df_csv = []

for label in data_labels:
    label_files_directory = os.listdir(data_directory + label)
    for file in label_files_directory:
        file_path = data_directory + label + "/" + file
        df_train_summary, df_test_summary = create_images(file_path, label, file)
        train_df_csv.append(df_train_summary)
        test_df_csv.append(df_test_summary)

train_df_csv = pd.concat(train_df_csv)
test_df_csv = pd.concat(test_df_csv)
train_df_csv.to_csv('train.csv', index=False)
test_df_csv.to_csv('test.csv', index=False)