import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def dec(func):
    def wrapper(*args, **kwargs):
        print("\n")
        print(f"Model starts learning from the data")
        result = func(*args, **kwargs)
        return result
    return wrapper

# Hyper-parameters
batch_size = 64
num_classes = 10
learning_rate = 0.001
input_size = 1
num_epochs = 3

# create an dataset
train_dataset = datasets.MNIST(
    root="datasets/",
    download=True,
    train=False,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root="datasets/",
    train=False,
    download=False,
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

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super(CNNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the array
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# set the device agnostic model
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialixe the network
cnn_model = CNNNetwork(input_channels=input_size, num_classes=num_classes).to(device)

# create the loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params= cnn_model.parameters(), lr=learning_rate)

# create an training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_dataloader)):
        # move the data to the appropirate device
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        scores = cnn_model(data)
        loss = criterion(scores, targets)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

# create an evaluation loop
@dec
def check_accuracy(loader: DataLoader, model) -> float:
    num_correct = 0
    num_samples = 0

    # set the model into inference mode
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

# test the functionalities
train = check_accuracy(train_dataloader, cnn_model)
test = check_accuracy(test_dataLoader, cnn_model)
print(f"Accuracy on training set: {train*100:.2f}")
print(f"Accuracy on testing set: {test*100:.2f}")