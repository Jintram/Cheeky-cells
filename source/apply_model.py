
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
import matplotlib.pyplot as plt

sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/2025_Cheeky-cells/')
import source.preprocessing as cheepre

import mypytorch.mymodels as mm

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
    

def loadmodel():
    
    PATH='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/pytorchmodel_lr.1_66pct_2025-04-09_14-33.pth'
    PATH='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/pytorchmodel_verySimple-78pct_2025-04-10_17-48.pth'    
    model = mm.VerySimpleNN().to("mps")
    
    PATH='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/pytorchmodel_CNN_90pct_2025-04-10_18-16.pth'
    model = mm.CNN((1, 29, 29),6).to("mps")
    
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    # sum(p.numel() for p in model.parameters())
    
    return model

def loadimage():
    
    # arbitrary iamge
    filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/resizedTo20_2x Cellen 3 (20x vegroting).jpg'
    # image that was trained on
    filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/resizedTo20_Cellen tellen, 10x foto 1.jpg'
    filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/resizedTo20_10x viv wang mb 2.jpg'
    img = Image.open(filepath)
    img = np.array(img)
    
    return img    
    
def applymodeltoimage_batchmode(img):
    SIZE_HALF=14
    
    image_sizei = img.shape[0]
    image_sizej = img.shape[1]

    start_i = SIZE_HALF+1
    start_j = SIZE_HALF+1
    end_i   = image_sizei - SIZE_HALF
    end_j   = image_sizej - SIZE_HALF

    output_img = np.zeros_like(img, dtype=np.int8)
    #img_collection_tensor = torch.zeros((end_i-start_i)*(end_j-start_j), 1, 29, 29, requires_grad=False).to("mps")
    img_collection_array  = np.zeros([(end_i-start_i)*(end_j-start_j), 1, 29, 29], dtype=np.uint8)
    
    print('Preparing image')
    for idx_i in range(start_i, end_i):
        for idx_j in range(start_j, end_j):
            # idx_i = SIZE_HALF+1; idx_j = SIZE_HALF+1
            
            # define the square
            square = img[(idx_i-SIZE_HALF):(idx_i+SIZE_HALF+1), (idx_j-SIZE_HALF):(idx_j+SIZE_HALF+1)].astype(np.uint8)
            square_exp = np.expand_dims(square, axis=0) # expects shape [1, 29, 29]
            
            # convert to tensor
            # This requires insane amounts of memory for larger objects
            # Need to do this later on the fly
            # square_tensor = torch.from_numpy(square_exp).to("mps")
                # plt.imshow(square); plt.show(); plt.close()
            
            # add tensor to collection
            idx = (idx_i-start_i)*(end_j-start_j) + (idx_j-start_j)
            img_collection_array[idx] = square_exp
        
        if idx_i % 100 == 0:
             print(f'{idx_i}/{end_i} lines done')
           
    print('Creating predictions')
                            
    # apply model at once
    # pred_img = model(img_collection_tensor)
    
    # build up the prediction image in batches of 1000
    batch_size = 1000
    num_batches = int(np.ceil(img_collection_array.shape[0] / batch_size))    
    # flattened pred image with one-hot coding
    pred_img = np.zeros([img_collection_array.shape[0], 6])
    # pred_img = torch.zeros((img_collection_tensor.shape[0], 6), requires_grad=False).to("mps")
    for batch_idx in range(num_batches):
        # batch_idx = 0
        # batch_idx += 1
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, img_collection_array.shape[0])
        
        # apply model
        current_input = torch.from_numpy(img_collection_array[start_idx:end_idx].astype(np.float32)).to("mps")
        current_pred = model(current_input)
        
        # save it to applicable location
        # conversion to numpy necessary, otherwise memory heavy (crash)
        pred_img[start_idx:end_idx] = current_pred.cpu().detach().numpy()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx+1}/{num_batches} done')
    
    # apply argmax
    pred_img_int = np.argmax(pred_img, axis=1)
    
    # use reshape to get same dimensions as original image
    output_img = pred_img_int.reshape((end_i-start_i), (end_j-start_j))
    
    
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
    
    
######### OLD


def applymodeltoimageOLD(img, SIZE_HALF=14):
    
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
    