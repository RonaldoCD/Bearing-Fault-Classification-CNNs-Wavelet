import pandas as pd
from skimage import io, transform

from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, train=True, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.wavelet_df = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.wavelet_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.wavelet_df.iloc[idx, 0])
        image = image[:, :, 0:3]
        label = self.wavelet_df.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        if self.train:
            return image, label
        else:
            return image


class WaveletNet(nn.Module):
    def __init__(self):
        super(WaveletNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 8, (3, 5), (1, 2))
        # self.conv2 = nn.Conv2d(8, 16, (3, 5), (1, 2))
        # self.conv3 = nn.Conv2d(16, 32, 3, 1)

        self.conv1 = nn.Conv2d(3, 8, (3, 5), (1, 2))
        self.conv2 = nn.Conv2d(8, 16, (3, 5), (1, 2))
        self.conv3 = nn.Conv2d(16, 32, 3, 1)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

        self.conv2_bn = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(0.5)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_bn(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Run max pooling over x
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through ``fc1``
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        # x = self.dropout(x)
        out = self.fc3(x)
        return out