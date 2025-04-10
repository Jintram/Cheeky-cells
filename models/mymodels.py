
# I will define more sophisticated models here

import torch
from torch import nn
from tqdm import tqdm_notebook

class CNN(nn.Module):
    
    # This model comes from
    # https://gist.github.com/devanshuDesai
    # See devanshuDesai/mnist_dodgers.py
    # https://gist.github.com/devanshuDesai/9f06681d8939afd04f8fab5ac5f5dbf8
    
    # This model can take multiple input sizes,
    # to take my 29x29 input block, set 
    # input_size: (1,28,28)
    # classes can be set to 5 (or 6)
    
    def __init__(self, input_size, num_classes):
        """
        init convolution and activation layers
        Args:
            input_size: (1,28,28)
            num_classes: 10
        """
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(4 * 4 * 64, num_classes)
        
    
    def forward(self, x):
        """
        forward function describes how input tensor is transformed to output tensor
        Args:
            x: (Nx1x28x28) tensor
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x
 
def examplecoderemove():
    
    modelCNN = CNN((1, 29, 29), 5)    
        # sum(p.numel() for p in modelCNN.parameters())
        
    opts = {
        'lr': 1e-3,
        'epochs': 20,
        'batch_size': 64
    }

    optimizer = torch.optim.Adam(model.parameters(), opts['lr'])
    criterion = torch.nn.CrossEntropyLoss()  # loss function



