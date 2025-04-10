# Load applicable library
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image

import numpy as np

import sys
from matplotlib.colors import ListedColormap
sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/2025_Cheeky-cells/')
import source.preprocessing as cheepre

import seaborn as sns

cm_to_inch = 1/2.54

# add bang_wong_colors vector
color_palette = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000"   # Black
] 


# define my model
# (note: carefull! this is redundant, may need to centralize this!)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(29*29, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

def loadmodel():
    
    PATH='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/pytorchmodel_lr.1_66pct_2025-04-09_14-33.pth'
    
    model = NeuralNetwork().to("mps")
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    # sum(p.numel() for p in model.parameters())
    
    return model

def loadimage():
    
    filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/resizedTo20_2x Cellen 3 (20x vegroting).jpg'
    
    img = Image.open(filepath)
    img = np.array(img)
    
    return img    
    
def applymodeltoimage(img, SIZE_HALF=14):
    
    image_sizei = img.shape[0]
    image_sizej = img.shape[1]

    start_i = SIZE_HALF+1
    start_j = SIZE_HALF+1
    end_i   = image_sizei - SIZE_HALF
    end_j   = image_sizej - SIZE_HALF

    output_img = np.zeros_like(img, dtype=np.int8)
    for idx_i in range(start_i, end_i):
        for idx_j in range(start_j, end_j):
            # idx_i = SIZE_HALF+1; idx_j = SIZE_HALF+1
            
            # define the square
            square = img[(idx_i-SIZE_HALF):(idx_i+SIZE_HALF+1), (idx_j-SIZE_HALF):(idx_j+SIZE_HALF+1)].astype(np.float32)
            square_exp = np.expand_dims(square, axis=0) # expects shape [1, 29, 29]
            
            # convert to tensor
            square_tensor = torch.from_numpy(square_exp).to("mps")
                # plt.imshow(square); plt.show(); plt.close()
                        
            # apply model
            pred = model(square_tensor)
            # get the index of the max value
            pred_idx = torch.argmax(pred, dim=1)
            # convert to python integer
            pred_idx = int(pred_idx.item())
            
            output_img[idx_i, idx_j] = pred_idx
            
        if ((idx_i) % 10) == 0:
            print(f'{idx_i}/{end_i} lines done ({round(idx_i/end_i*100,1)}%).')
            unique, counts = np.unique(output_img, return_counts=True)
            print(counts)
    
    
def showsidebyside():
        
    # convert color_palette to cmap
    cmap = ListedColormap(color_palette)
    
    thecategories = ['unclassified', 'background', 'cell border', 'intercell border', 'cell cytoplasm', 'nuclei']
    
    theunique, thecounts = np.unique(output_img, return_counts=True)
    categories_as_int = np.arange(len(thecategories))
    thebins = np.linspace(0, len(thecategories), len(thecategories)+1)-.5
    thebincounts, _ = np.histogram(output_img, bins=thebins)

    fig, ax = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))
    ax[0].imshow(img)
    ax[1].imshow(output_img, cmap=cmap, vmin=0, vmax=np.max(categories_as_int)+1)
    plt.show(); plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    sns.barplot(x=categories_as_int, y=thebincounts, ax=ax, palette=color_palette)
    for idx in range(len(thecategories)):
        ax.text(idx, max(thecounts)/100, str(thecategories[idx]), color='black', ha='center', va='bottom', rotation=90)
    ax.tick_params(axis='x', rotation=90)    
    ax.set_xlabel('Category'); ax.set_ylabel('Times observed')
    plt.tight_layout()
    plt.show(); plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    sns.barplot(y=categories_as_int, x=thebincounts, ax=ax, palette=color_palette, orient='h')
    for idx in range(len(thecategories)):
        ax.text(max(thebincounts)/100, idx, str(thecategories[idx]), color='black', ha='left', va='center', rotation=0)
    ax.tick_params(axis='y', rotation=0)    
    ax.set_ylabel('Category'); ax.set_xlabel('Times observed')
    # set log scale for x
    ax.set_xscale('log')
    plt.tight_layout()
    plt.show(); plt.close()
    
    # histogram of the unique values in output_img
    
    counts
    