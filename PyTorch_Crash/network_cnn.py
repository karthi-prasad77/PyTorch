import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# parameters
batch_size = 64

# create an dataset
train_dataset = datasets.MNIST(
    root="datasets/",
    download=True,
    train=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root="datasets/",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# create an dataloader for easy data management
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_dataLoader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)

# build an cnn model
class CNNNetwork(nn.Module):

    def __init__(self, input_channels, num_classes):
        super(CNNNetwork, self).__init__()


    def forward(self, x):
        pass


        