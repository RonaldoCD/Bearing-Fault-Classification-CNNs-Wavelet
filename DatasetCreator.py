import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import os

IMAGES_DIRECTORY = "/home/ronaldocd/Desktop/RC/Wavelet/TESIS/IMAGENES/"
N_IMAGES_TRAINING_SET = 243 # number of images to create
N_IMAGES_TEST_SET = 41
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

    #
    # df = df.iloc[:, [1,2,3]]
    #
    # df = pd.DataFrame(df_scaled, columns=[
    #     'x', 'y', 'z'])

    # print("   ")
    print("Label: ", label)
    print("File: ", file)
    # print(df.describe())

    random_idxs = np.random.randint(low=0, high=df.shape[0] - 500, size=N_IMAGES)
    random_idxs_train = random_idxs[:N_IMAGES_TRAINING_SET]
    random_idxs_test = random_idxs[N_IMAGES_TRAINING_SET:]
    # signal = df[AXIS].values
    signal = df[AXIS] * CALIB_FACTOR

    signal = (signal - np.mean(signal)) / np.std(signal)
    # print("Signal mean: ", np.mean(signal))
    # print("Signal std: ", np.std(signal))

    widths = np.arange(1, SCALE)

    for n, i in enumerate(random_idxs_train):
        cwtmatr, freqs = pywt.cwt(signal[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)
        # print("CWT mean: ", np.mean(cwtmatr))
        # print("CWT std: ", np.std(cwtmatr))
        image_name = label + "_" + file[:-4] + "_" + str(n) + ".png"
        # image_name_2 = label + "_" + file[:-4] + "_" + str(n) + "pil.png"

        image_path = IMAGES_DIRECTORY + "train" + "/" + label + "/" + image_name
        # image_path = IMAGES_DIRECTORY + "train" + "/" + label + "_" + AXIS + "/" + image_name

        plt.imsave(image_path, cwtmatr, cmap='gray', vmax=abs(cwtmatr).max(),
                   vmin=-abs(cwtmatr).max())

        # cwt = (((cwtmatr - cwtmatr.min()) / (cwtmatr.max() - cwtmatr.min())) * 255.9).astype(np.uint8)
        # img = Image.fromarray(cwt)
        # img.save(image_name_2)

        plt.imsave(image_path, cwtmatr, cmap='gray', vmax=abs(cwtmatr).max(),
                   vmin=-abs(cwtmatr).max())

        # plt.imsave(image_path, cwtmatr, cmap='PRGn', vmax=np.std(cwtmatr), vmin=-np.std(cwtmatr))

        new_row_train = {'image_path': image_path, 'label':label, 'label_idx':label[6]}
        df_train_summary.loc[len(df_train_summary)] = new_row_train
        # Use the loc method to add the new row to the DataFrame
        # print(freqs)

    for n, i in enumerate(random_idxs_test):
        cwtmatr, freqs = pywt.cwt(signal[i:i + N_SAMPLES], widths, 'morl', sampling_period=sampling_period)
        # cwtmatr = abs(cwtmatr)

        # cwtmatr_x, freqs = pywt.cwt(signal_x[i:i + N_SAMPLES], widths, 'shan2.0-0.1', sampling_period=sampling_period)
        # cwtmatr_y, freqs = pywt.cwt(signal_y[i:i + N_SAMPLES], widths, 'shan2.0-0.1', sampling_period=sampling_period)
        # cwtmatr_z, freqs = pywt.cwt(signal_z[i:i + N_SAMPLES], widths, 'shan2.0-0.1', sampling_period=sampling_period)
        # cwtmatr_x = abs(cwtmatr_x)
        # cwtmatr_y = abs(cwtmatr_y)
        # cwtmatr_z = abs(cwtmatr_z)
        #
        # cwtmatr = (cwtmatr_x + cwtmatr_y + cwtmatr_z) / 3

        image_name = label + "_" + file[:-4] + "_" + str(n) + ".png"
        image_path = IMAGES_DIRECTORY + "test" + "/" + label + "/" + image_name
        # image_path = IMAGES_DIRECTORY + "test" + "/" + label + "_" + AXIS + "/" + image_name

        plt.imsave(image_path, cwtmatr, cmap='gray', vmax=abs(cwtmatr).max(),
                   vmin=-abs(cwtmatr).max())
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