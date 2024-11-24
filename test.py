import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import NeuralNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784  # 28x28 flattened to 784 1D tensor
hidden_size = 1000
num_classes = 10  # 0-9
batch_size = 100

model = NeuralNet(input_size, hidden_size, num_classes)

# Load the model
FILE = "model.pth"
model.load_state_dict(torch.load(FILE, weights_only=True))
model.to(device)
model.eval()

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())


test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)  # number of elements in the tensor
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
