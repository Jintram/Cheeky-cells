
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
            nn.MaxPool2d(kernel_size=2, padding=1)) # added padding = 1 
            # remaining plane dimension = np.floor( ((N-4)+1) /2 )
            # N=28 --> (28-4+1)//2 --> 12
            # N=29 --> (29-4+1)//2 --> 13
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1)) # added padding = 1
            # remaining plane dimension = np.floor( ((N-4)+1) /2 )
            # N=28; (12-4+1)//2 --> 4
            # N=29; (13-4+1)//2 --> 5

        # calculate how big the resulting plane is after the convs and pools
        # adapted code MW
        remaining_size = int(  (((input_size[1] - 3)//2)-3)//2  )
        self.fc1 = nn.Linear(remaining_size * remaining_size * 64, num_classes)
        
        # self.fc1 = nn.Linear(4 * 4 * 64, num_classes)
        
    
    def forward(self, x):
        """
        forward function describes how input tensor is transformed to output tensor
        Args:
            x: (Nx1x28x28) tensor
        """
        x = self.layer1(x)
        #print('1', x.shape)
        x = self.layer2(x)
        #print('2', x.shape)
        x = x.reshape(x.size(0), -1)
        #print('3 (flat??)', x.shape)
        x = self.fc1(x)
        #print('4', x.shape)
        return x
 

def testsomestuff0():

    modelCNN = CNN((1, 29, 29),6).to("mps")
        # sum(p.numel() for p in modelCNN.parameters())
        
    X2 = torch.rand(1, 1, 29, 29, device="mps") # random image as test input
    logits = modelCNN(X2) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
    logits
    
    # test batch input
    X3 = torch.rand(64, 1, 29, 29, device="mps") # random image as test input
    logits = modelCNN(X3) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
    logits

################################################################################
# original model

class CNNoriginal(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        init convolution and activation layers
        Args:
            input_size: (1,28,28)
            num_classes: 10
        """
        
        super(CNNoriginal, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=5),
                # note input_size[0] allows for multi-channel input
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
        #print('1', x.shape)
        x = self.layer2(x)
        #print('2', x.shape)
        x = x.reshape(x.size(0), -1)
        # x = x.flatten()
        #print('3 (flat?)', x.shape)
        x = self.fc1(x)
        #print('4', x.shape)
        return x
    

def testsomestuff():

    modelCNNorginial = CNNoriginal((1, 28, 28),6).to("mps")
        # sum(p.numel() for p in modelCNN.parameters())
        
    X2 = torch.rand(1, 1, 28, 28, device="mps") # random image as test input
    logits = modelCNNorginial(X2) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
    logits
    
    # test batch input
    X3 = torch.rand(64, 1, 28, 28, device="mps") # random image as test input
    logits = modelCNNorginial(X3) # "logits" refers to raw, unnormalized outputs of the final layer of a neural network
    logits