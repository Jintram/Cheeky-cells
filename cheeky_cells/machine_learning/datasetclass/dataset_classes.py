
# %% ################################################################################

import sys
import os

import torch

import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor, Lambda
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms

import numpy as np

from PIL import Image

import warnings

import readwrite.cheeky_readwrite as crw
    # import importlib; importlib.reload(crw)

# custom code
# import mypytorch_fullseg.mymodels_fullseg as mm
# import mypytorch_fullseg.dataset_classes_fullseg as md

# reload the custom libs
# import importlib; importlib.reload(mm); importlib.reload(md); importlib.reload(cheepre)

# Based on tutorial and notebook listed below
# https://pytorch.org/tutorials/beginner/basics/intro.html
# /Users/m.wehrens/Documents/git_repos/_UVA/2025_MW-testing-ML/simple-tutorial.ipynb


# %% ################################################################################

def get_label_frequencies(dataset, suffix_img, suffix_lbl):
    '''
    Determines the label frequencies based on file list in 
    dataset class ImageDataset_tiles.filelist_imgs.    
    '''
    
    # get paths for all files in dataset
    allfilepaths = [dataset.datadir + filename for filename in dataset.filelist_imgs]
    
    # loop over all training data, count labels
    stats_all = [np.bincount(np.load(filepath.replace(suffix_img, suffix_lbl), allow_pickle=True).flatten())
                 for filepath in allfilepaths]    
    
    # now sum
    nr_labels = np.max([len(X) for X in stats_all])
    stats_all_zeropadded = [np.append(X, [0]*(nr_labels-len(X))) for X in stats_all]
    
    # now determine counts
    label_counts = np.sum(np.array(stats_all_zeropadded), axis=0)
    
    return label_counts
    
def get_label_weights(dataset, suffix_img, suffix_lbl, device='mps'):
    '''
    Determine label weights based on frequencies in dataset.
    Based on file list in dataset class ImageDataset_tiles.filelist_imgs. 
    Returns a tensor on "device".
    '''
    
    label_counts = get_label_frequencies(dataset, suffix_img, suffix_lbl)
    label_weights = torch.tensor(1/(label_counts/np.sum(label_counts)), dtype=torch.float32).to(device)
    
    return label_weights

class ImageDataset_tiles(Dataset):
    '''    
    This dataset class is aimed to load ±2000x2000 images, and 
    using a transformer, should then respond with 1000x1000 crops
    that are randomly selected/transformed as data output.
    (Note: these resolutions can be changed.)
    
    Images and labels are assumed to be .npy format, stored
    in the datadir, with filename as listed in df_metadata,
    and suffix as specified by user.
    
    INPUT PARAMETERS:
    - datadir: directory with training data and label files
    - df_metadata: pandas df with metadata, with minimally 
    the columns 'filename' and 'train_or_test'.
    - train_or_test: set to 'training' or 'test', to generate
    respective data sets.    
    - ARTIFICIAL_N: including augmentation, how many samples it
    will return.
    - CROP_SIZE: size of cropping square, default 1000.
    (To do: not sure cropping is necessary, as it's a fully conv
    network.)
    
    Since the transformer does that latter part, this dataset
    should just load the data and the labels.
    
    Input training data will come from pre-processed .npy files,
    which contain the image, the segmentation mask, and 
    potentially another image that is pre-processed to highlight
    specific features.
    '''
    
    def __init__(self, df_metadata, datadir, train_or_test, 
                 img_suffix='_img', lbl_suffix='_seg_postpr', # img_suffix='_tile_img'; lbl_suffix='_tile_seg_postpr'
                 transform=None, transform_label=None, 
                 targetdevice="mps", ARTIFICIAL_N=1000, CROP_SIZE=1000):#, add_dim=False):
        
        # First get lists of all the files that are required
        df_metadata_sel = df_metadata.loc[df_metadata['train_or_test'] == train_or_test].reset_index(drop=True)   
        self.filelist_imgs   = crw.addsuffixtofilenames(df_metadata_sel['filename'].tolist(), suffix=img_suffix)
        self.filelist_labels = crw.addsuffixtofilenames(df_metadata_sel['filename'].tolist(), suffix=lbl_suffix)
        
        # directory and device
        self.datadir = datadir
        self.targetdevice = targetdevice
        
        # transforms        
        self.transform = transform
        self.transform_label = transform_label
        
        # amount of actual samples
        self.num_samples = len(self.filelist_imgs)
        # amount of samples it will say it have, set to SIZE_ARTIFICIAL unless more actual samples 
        self.len_augment = max(len(self.filelist_imgs), ARTIFICIAL_N)

    def __len__(self):
        return self.len_augment

    def __getitem__(self, idx):
        
        # idx will run higher than actual nr of samples thanks to augmentation
        # but to keep sampling underlying samples evenly, use modulo
        sample_idx = idx % self.num_samples
        
        # now produce the label and the image
        image   = np.load(self.datadir + self.filelist_imgs[sample_idx], allow_pickle=True)
        label   = np.load(self.datadir + self.filelist_labels[sample_idx], allow_pickle=True)
        
        # transform
        if self.transform:
            # apply same transformation to both input and annotation
            shared_seed = torch.randint(0, 4294967296, (1,)).item() # 4294967296 = 2**32
            
            torch.manual_seed(shared_seed)
            image = self.transform(image)
            torch.manual_seed(shared_seed)
            label = self.transform_label(label)            
        else:
            warnings.warn("Transform not set, returning original image!", UserWarning, stacklevel=2)   
            # convert the image to a torch object
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)                
            
        return image.to(self.targetdevice), label.to(self.targetdevice)



# additionally, define some transforms to apply to the images
# Define the augmentation pipeline
augmentation_pipeline_input = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)), # convert to PIL image    
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
    transforms.RandomRotation(degrees=45),  # Random rotation within ±45 degrees    
    transforms.RandomCrop((500, 500)),  # Randomly crop 500x500 patches
        # note: crop could be done first to speed things up, but then cropping results
        # results in many sliced cells with edges that are not annotated ideally.
    transforms.ToTensor(),  # Convert to PyTorch tensor
    # could add some normalization if desired    
    # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (example for grayscale)
])

augmentation_pipeline_label = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)), # convert to PIL image    
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
    transforms.RandomRotation(degrees=45),  # Random rotation within ±45 degrees    
    transforms.RandomCrop((500, 500)),  # Randomly crop 500x500 patches
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))  # Convert to LongTensor
])

# %%

# %% ################################################################################