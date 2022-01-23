import torch
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.detection import transform
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import gzip
import tarfile
import lzma
from PIL import Image



# Part 0 - Boilerplate/variable declaration


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Allows PyTorch to use Either CUDA (Nvidia GPU) or CPU to train the model
print(f'Using {device} device')

Transform = transforms.Compose(
    [transforms.Resize(255),  # Makes all images a 255x255 size
     transforms.CenterCrop(224),  # crops the images around the centre
     transforms.ToTensor(),  # Converts PIL images to PyTorch Tensors
     transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))])  # Normalises the tensors using their mean and standard deviation

batch_size = 4

# Part 1 - DataLoading






TrainingSet = datasets.Caltech256(root="C:\\Users\\griff\\PycharmProjects\\CaltechModel", transform=Transform,
                                  download=True)
TrainingLoader = DataLoader(TrainingSet, batch_size=batch_size, shuffle=True, num_workers=2)

TestSet = datasets.Caltech101(root="C:\\Users\\griff\\PycharmProjects\\CaltechModel", transform=Transform,
                              download=True)
TestLoader = DataLoader(TestSet, batch_size=batch_size, shuffle=True, num_workers=2)




#Part 2 - CNN


def main():

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(TrainingLoader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            print('hi')
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            print('hi')
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            print(x.shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(TrainingLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './Caltech256.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()
