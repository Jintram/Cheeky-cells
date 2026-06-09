


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
         
def get_label_frequencies_train(metadata_file, annot_dir):
    # annot_dir='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
    # metadata_file='/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/metadata_Fluoppi_data20250328_MACHINELEARN.xlsx'
        
    thefilelist_imgs, \
    thefilelist_annot, \
    thefilelist_extra, \
    thefilelist_enhanced, \
    thefilelist_annot_features = \
        get_file_list_annotimgs_dloader(metadata_file, 'train')
    
    counts_all=np.array([0]*4)
    for sample_idx in range(len(thefilelist_enhanced)):
        # sample_idx=0
        # load image
        current_annot = np.load(annot_dir + thefilelist_annot_features[sample_idx], allow_pickle=True)
        # count frequency of values in the image
        unique, counts = np.unique(current_annot, return_counts=True)
        counts_all += counts
    
    return (counts_all)
    

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
                 targetdevice="mps", SIZE_ARTIFICIAL=1000):#, add_dim=False):
        
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
        
        # amount of actual samples
        self.num_samples = len(self.thefilelist_imgs)
        # amount of samples it will say it have, set to SIZE_ARTIFICIAL unless more actual samples 
        self.len_augment = max(len(self.thefilelist_imgs), SIZE_ARTIFICIAL)

    def __len__(self):
        return self.len_augment

    def __getitem__(self, idx):
        
        # idx will run higher than actual nr of samples thanks to augmentation
        # but to keep sampling underlying samples evenly, use modulo
        sample_idx = idx % self.num_samples
        
        # now produce the label and the image
        image   = np.load(self.annot_dir + self.thefilelist_enhanced[sample_idx], allow_pickle=True)
        label   = np.load(self.annot_dir + self.thefilelist_annot_features[sample_idx], allow_pickle=True)
        
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

augmentation_pipeline_target = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)), # convert to PIL image    
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
    transforms.RandomRotation(degrees=45),  # Random rotation within ±45 degrees    
    transforms.RandomCrop((500, 500)),  # Randomly crop 500x500 patches
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))  # Convert to LongTensor
])
