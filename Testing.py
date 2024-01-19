import NN_Classes
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch
import numpy as np

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

test_csv_file = "test.csv"

loss_fn = torch.nn.CrossEntropyLoss()
model = NN_Classes.WaveletNet()
if train_on_gpu:
    model.cuda()
model.load_state_dict(torch.load('model_wavelet.pt'))
model.eval()

# Image Transforms
image_transform = transforms.Compose([
    transforms.ToTensor()
])


test_dataset = NN_Classes.WaveletDataset(test_csv_file, train=True, transform=image_transform)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
test_loss = 0
test_accuracy = 0
with torch.no_grad():
    for data, target in test_dataloader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = loss_fn(output, target)
        # update average validation loss
        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        batch_valid_accuracy = np.sum(correct)
        test_accuracy += batch_valid_accuracy

test_accuracy /= len(test_dataset)
test_loss = test_loss / len(test_dataset)

print("  Testing Accuracy: {}".format(test_accuracy))
print("  Test Loss: {}".format(test_loss))



