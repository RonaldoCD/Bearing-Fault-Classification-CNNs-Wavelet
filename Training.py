import NN_Classes
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch
import numpy as np


# CSV Files
train_csv_file = "train.csv"
test_csv_file = "test.csv"

# Image Transforms
image_transform = transforms.Compose([
    transforms.ToTensor()
])

# Creating dataset
wavelet_dataset = NN_Classes.WaveletDataset(train_csv_file, train=True, transform=image_transform)

# Splitting dataset in train and validation datasets
train_size = int(0.8 * len(wavelet_dataset))
valid_size = len(wavelet_dataset) - train_size
train_dataset, valid_dataset = random_split(wavelet_dataset, [train_size, valid_size])

# Creating train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

print("Train data loader: ", len(train_dataset))
# Loss function

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# Neural network
model = NN_Classes.WaveletNet()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

loss_fn = torch.nn.CrossEntropyLoss()

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# number of epochs to train the model
n_epochs = 20

best_valid_acc = - np.Inf  # track change in validation loss

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_dataloader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = loss_fn(output, target)
        # print("Conv 1 grads before: ", model.conv1.weight.grad)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # print("Conv 1 grads after: ", model.conv1.weight.grad)
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)

    ######################
    # validate the model #
    ######################
    model.eval()
    valid_accuracy = 0.0
    with torch.no_grad():
        for data, target in valid_dataloader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = loss_fn(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            batch_valid_accuracy = np.sum(correct)
            valid_accuracy += batch_valid_accuracy

    valid_accuracy /= len(valid_dataset)
    # calculate average losses
    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(valid_dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    print("  Validation Accuracy: {}".format(valid_accuracy))

    # save model if validation loss has decreased
    if valid_accuracy > best_valid_acc:
        print('  Validation accuracy increased ({:.3f} --> {:.3f}).  Saving model ...'.format(
            best_valid_acc,
            valid_accuracy))
        print(" ")
        torch.save(model.state_dict(), 'model_wavelet.pt')
        best_valid_acc = valid_accuracy

