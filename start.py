import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Basic Neural Network for Classification
class PerceptionNet(nn.Module):
    def __init__(self):
        super(PerceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Placeholder for Generative Network (mimicking and modifying previous outputs)
class GenerativeNet(nn.Module):
    def __init__(self):
        super(GenerativeNet, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Setup Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

trainset = datasets.FakeData(transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

def train_network(net, dataloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Instantiate and train the first network
perception_net = PerceptionNet()
train_network(perception_net, trainloader)

# Generate pseudo data from the first network
pseudo_data = []
pseudo_labels = []
for inputs, _ in trainloader:
    with torch.no_grad():
        outputs = perception_net(inputs)
        pseudo_data.append(outputs)
        pseudo_labels.append(torch.argmax(outputs, dim=1))

pseudo_data = torch.cat(pseudo_data)
pseudo_labels = torch.cat(pseudo_labels)

pseudo_dataset = TensorDataset(pseudo_data, pseudo_labels)
pseudo_loader = DataLoader(pseudo_dataset, batch_size=32, shuffle=True)

# Instantiate and train the second network on generated data
generative_net = GenerativeNet()
train_network(generative_net, pseudo_loader)

print("First and second networks trained successfully.")
