import torch
import torch.nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# hyperparameters
batch_size = 5
learning_rate = 0.001
num_epoch = 8


# load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# build the CNN model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.batchnorm1 = torch.nn.BatchNorm2d(6)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5, padding = 2)
        self.fc1 = torch.nn.Linear(32 * 5 * 5, 300)
        self.batchnorm2 = torch.nn.BatchNorm1d(300)
        self.fc2 = torch.nn.Linear(300, 100)
        self.fc3 = torch.nn.Linear(100, 10) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.pool(F.relu(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.fc1(x))
        x = self.batchnorm2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()

# initialize cost function, optimizer, and scheduler
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

# evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images
        labels = labels
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
