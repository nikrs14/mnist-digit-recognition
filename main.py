import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ConvNet import ConvNet
from DigitsDataset import DigitsDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

num_epochs = 10
batch_size = 100
learning_rate = 0.001

personal_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor(), download = False)
personal_dataset = DigitsDataset('./data.csv', './images/', transform = personal_transform)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
personal_loader = DataLoader(personal_dataset)

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
      print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{n_total_steps}, Loss: {loss.item()}')
print('Finished Training')

with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

  acc = 100.0 * n_correct/n_samples
  print(f'Accuracy : {acc}%')

with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for image, labels in personal_loader:
    image = image.to(device)
    labels = labels.to(device)
    outputs = model(image)
    m = nn.Softmax(dim=1)

    _, predicted = torch.max(outputs, 1)
    print(f'Correct Answer: {labels.item()} | Prediction: {predicted.item()}')
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()

  acc = 100.0 * n_correct/n_samples
  print(f'Accuracy : {acc}%')
