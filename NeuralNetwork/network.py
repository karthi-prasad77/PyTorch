# import the libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from tqdm import tqdm

# create an fully connected network
class NeuralNetwork(nn.Module):
	def __init__(self, inputSize, numClasses):
		super(NeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(inputSize, 50)
		self.fc2 = nn.Linear(50, numClasses)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

# create an object for the neural network model
#net = NeuralNetwork(784, 10)  # 784 data with 10 classes
#x = torch.randn(64, 784)  # 32x32 = 64
#print(net(x).shape)

# set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
inputSize = 784
numClasses = 10
learningRate = 0.001
batchSize = 64
numEpochs = 3

print("Dataset download starts....")

# load the dataset
trainData = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)

testData = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

print("Dataset download finished !!")
print()
print("Dataset loaded to the memory")

# create an dataset loader
trainLoader = DataLoader(dataset = trainData, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(dataset = testData, batch_size=batchSize, shuffle=True)

# initialize the neural network
# process the data using the GPU
net = NeuralNetwork(inputSize=inputSize, numClasses=numClasses).to(device)

print(net.parameters())
# optimizer and criterion
optimizer = optim.Adam(net.parameters(), lr=learningRate)
criterion = nn.CrossEntropyLoss()

# training loop
for epoch in range(numEpochs): # set to the training mode
	for batchIdx, (data, targets) in enumerate(tqdm(trainLoader)):
		# get the data to the cuda
		data = data.to(device=device)
		targets = targets.to(device=device)

		# get the data to the correct shape
		data = data.reshape(data.shape[0], -1)
		# print("Sample data:", data)

		# forward method
		scores = net(data)
		loss = criterion(scores, targets)

		# backward propagation
		optimizer.zero_grad()
		loss.backward()

		optimizer.step()
	print()


# check the accuracy of the model
def checkAccuracy(loader, model):
	numCorrect = 0
	numSamples = 0

	# evaluate the model
	model.eval()

	with torch.no_grad():

		for x, y in loader:

			x = x.to(device=device)
			y = y.to(device=device)

			x = x.reshape(x.shape[0], -1)

			scores = model(x)

			_, prediction = scores.max(1)

			# print("Predictions : ", prediction)
			#print(_)

			numCorrect += (prediction == y).sum()

			numSamples += prediction.size(0)

			#print("Samples : ", numSamples)

	model.train()
	return numCorrect / numSamples

print(f"Accuracy on training set: {checkAccuracy(trainLoader, net)*100:.2f}")
print(f"Accuracy on test set: {checkAccuracy(testLoader, net)*100:.2f}")
