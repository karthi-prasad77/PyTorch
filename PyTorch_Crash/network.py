import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

# setting up the parameters
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784 # 28x28
num_classes = 10
learning_rate = 0.001
num_epochs = 3

# loading the standard dataset (MNIST)
train_dataset = datasets.MNIST(
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# create dataloader for easy data management
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)


class NeuralNetwok(nn.Module):

    def __init__(self, input_size: int, num_classes: int):
        super(NeuralNetwok, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# set the models optimizer and loss function
model = NeuralNetwok(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Model prameter : {model.parameters()}")

# build the training loop
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # push the data to the device
        data = data.to(device)
        targets = target.to(device)

        # convert the data into the specific shape
        data = data.reshape(data.shape[0], -1)

        # forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        # zero previous gradients
        optimizer.zero_grad()

        # back propagation
        loss.backward()

        optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy"
            f" {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()

check_accuracy(train_dataloader, model)
check_accuracy(test_dataloader, model)

#Note: CrossEntropy() loss would contains an softmax() function in it