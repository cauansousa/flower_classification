import torch, torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

#hyperparameters
input_size = (256, 256)
num_classes = 5
num_epochs = 120
batch_size = 128
lr_rate = 0.001

train_data = torchvision.datasets.ImageFolder('C:\\Users\\Cauan\\Documents\\algoritimos\\flowers\\train\\', transform=transforms.ToTensor())
test_data = torchvision.datasets.ImageFolder('C:\\Users\\Cauan\\Documents\\algoritimos\\flowers\\test\\', transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
net = net().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=lr_rate)
loss_function = nn.CrossEntropyLoss()

print("Starting training loop...")
for epoch in range(num_epochs):
    run_loss = 0.0
    for i, (input, labels) in enumerate(train_loader):
        
        input = input.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = net(input)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        
        run_loss += loss.item()
    print("Epoch: ", epoch+1,":", num_epochs, "// Loss: ", run_loss)
    
print('Finished Training')

#test model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(5)]
    n_class_samples = [0 for i in range(5)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(2):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {i} : {acc} %')
        
#save model
path = 'C:\\Users\\Cauan\\Documents\\GitHub\\Flower_classification\\flowers.pth'
torch.save(net.state_dict(), path)