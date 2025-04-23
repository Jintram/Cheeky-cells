


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

# custom code
# import mypytorch_fullseg.mymodels_fullseg as mm
# import mypytorch_fullseg.dataset_classes_fullseg as md

# reload the custom libs
# import importlib; importlib.reload(mm); importlib.reload(md); importlib.reload(cheepre)

# Based on tutorial and notebook listed below
# https://pytorch.org/tutorials/beginner/basics/intro.html
# /Users/m.wehrens/Documents/git_repos/_UVA/2025_MW-testing-ML/simple-tutorial.ipynb


def get_file_list_annotimgs_dloader(metadata_file, train_or_test):
    '''
    '''
    
    # Load metadata, get filenames
    metadata_table = pd.read_excel(metadata_file)
    bool_selection_metadata = metadata_table['train_or_test'] == train_or_test
    if bool_selection_metadata.sum() == 0:
        raise ValueError(f"No entries found for {train_or_test} in metadata file.")
    
    print(f'Selecting {bool_selection_metadata.sum()} samples for {train_or_test}')
    metadata_table_sel = metadata_table[bool_selection_metadata]
    
    # Get all sample names
    list_all_imgfiles_original = metadata_table_sel['filename'].values
    
    # And the corresponding saved annotated data files
    thefilelist_imgs =  [X.replace('.nd2', '_tile_img.npy') for X in list_all_imgfiles_original]
    thefilelist_annot = [X.replace('.nd2', '_tile_annothuman.npy') for X in list_all_imgfiles_original]
    thefilelist_extra = [X.replace('.nd2', '_tile_transform.npy') for X in list_all_imgfiles_original]
    thefilelist_enhanced = [X.replace('.nd2', '_tile_img_enhanced.npy') for X in list_all_imgfiles_original]
    thefilelist_annot_features = [X.replace('.nd2', '_tile_annothuman_features.npy') for X in list_all_imgfiles_original]
    
    return thefilelist_imgs, thefilelist_annot, thefilelist_extra, thefilelist_enhanced, thefilelist_annot_features
         

class ImageDataset_tiles(Dataset):
    '''
    This dataset class is aimed to load ±2000x2000 images, and 
    using a transformer, should then respond with 500x500 crops
    that are randomly selected/transformed as data output.
    
    Since the transformer does that latter part, this dataset
    should just load the data and the labels.
    
    Input training data will come from pre-processed .npy files,
    which contain the image, the segmentation mask, and 
    potentially another image that is pre-processed to highlight
    specific features.
    '''
    
    def __init__(self, annot_dir, metadata_file, train_or_test, transform=None, transform_target=None, 
                 targetdevice="mps"):#, add_dim=False):
        
        # first get lists of all the files that are required
        self.thefilelist_imgs, \
        self.thefilelist_annot, \
        self.thefilelist_extra, \
        self.thefilelist_enhanced, \
        self.thefilelist_annot_features = \
                get_file_list_annotimgs_dloader(metadata_file, train_or_test=train_or_test)
                # metadata_file='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/metadata_Fluoppi_data20250328_MACHINELEARN.xlsx'
        
        # directory and device
        self.annot_dir = annot_dir
        self.targetdevice = targetdevice
        
        # transforms        
        self.transform = transform
        self.transform_target = transform_target
        
        # amount of samples to return
        self.num_samples = len(self.thefilelist_imgs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        # now produce the label and the image
        image   = np.load(self.annot_dir + self.thefilelist_enhanced[idx], allow_pickle=True)
        label   = np.load(self.annot_dir + self.thefilelist_annot_features[idx], allow_pickle=True)
        
        # transform
        if self.transform:
            # apply same transformation to both input and annotation
            shared_seed = torch.randint(0, 4294967296, (1,)).item() # 4294967296 = 2**32
            
            torch.manual_seed(shared_seed)
            image = self.transform(image)
            torch.manual_seed(shared_seed)
            label = self.transform_target(label)
            
        return image.to(self.targetdevice), label.to(self.targetdevice)



# additionally, define some transforms to apply to the images
# Define the augmentation pipeline
augmentation_pipeline_input = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)), # convert to PIL image
    transforms.RandomCrop((500, 500)),  # Randomly crop 500x500 patches
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
    transforms.RandomRotation(degrees=45),  # Random rotation within ±45 degrees    
    transforms.ToTensor(),  # Convert to PyTorch tensor
    # could add some normalization if desired    
    # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (example for grayscale)
])

augmentation_pipeline_target = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)), # convert to PIL image
    transforms.RandomCrop((500, 500)),  # Randomly crop 500x500 patches
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
    transforms.RandomRotation(degrees=45),  # Random rotation within ±45 degrees    
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))  # Convert to LongTensor
])
