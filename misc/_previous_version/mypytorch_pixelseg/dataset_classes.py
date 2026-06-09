


import sys
import os

import torch

import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor, Lambda
from torch.optim.lr_scheduler import StepLR

import numpy as np

import source.preprocessing as cheepre

# custom code
import mypytorch_pixelseg.mymodels as mm
import mypytorch_pixelseg.dataset_classes as md

# reload the custom libs
# import importlib; importlib.reload(mm); importlib.reload(md); importlib.reload(cheepre)

# Based on tutorial and notebook listed below
# https://pytorch.org/tutorials/beginner/basics/intro.html
# /Users/m.wehrens/Documents/git_repos/_UVA/2025_MW-testing-ML/simple-tutorial.ipynb

class CustomImageDataset(Dataset):
    def __init__(self, annot_dir, img_dir, transform=None, target_transform=None, targetdevice="mps", preload_imgs=False, MYSEED=42):#, add_dim=False):
        
        self.annot_dir = annot_dir
        self.img_dir = img_dir
        self.targetdevice = targetdevice
        self.preload_imgs = preload_imgs
        # self.add_dim = add_dim # toggle to add extra dimension at start
            
        # get dataset info
        annot_pixelcount_list, list_allimgpaths, list_annotfilepaths = cheepre.acquire_trainingset_info(img_dir, annot_dir)
        self.annot_pixelcount_list = annot_pixelcount_list
        self.img_list = list_allimgpaths
        self.img_annot_list = list_annotfilepaths
        
        # get dataset img_idx, locx, locy, label
        self.img_idxs, self.posis, self.posjs, self.labels = \
            cheepre.build_labels_and_positions(annot_dir, annot_pixelcount_list, list_allimgpaths, list_annotfilepaths)
        self.labels = self.labels.astype(int)
        
        if self.preload_imgs:                 
            self.largeimgs = \
                {filename_img: cheepre.provide_img(img_dir, filename_img) for filename_img in list_allimgpaths}
                # get object size for the above dict
                # sys.getsizeof({filename_img: cheepre.provide_img(img_dir, filename_img) for filename_img in list_allimgpaths})
                
        # now manually shuffle the dataset
        # (I do this manually to also allow weighted sampling later)
        # (shuffle=True and weighed sampling are mutually exclusive)
        np.random.seed(MYSEED)
        shuffle_indices = np.arange(len(self.labels))
        np.random.shuffle(shuffle_indices)
        # # now shuffle all the information
        self.img_idxs = self.img_idxs[shuffle_indices]
        self.posis    = self.posis[shuffle_indices]
        self.posjs    = self.posjs[shuffle_indices]
        self.labels   = self.labels[shuffle_indices]
        
        # now determine weights 
        label_frequency = np.bincount(self.labels)
        weights_for_samples = 1.0 / label_frequency
        weights_for_samples[label_frequency==0] = 0 # counters div 0
        self.weights = weights_for_samples[self.labels]  # Assign a weight to each sample
        
        # transforms        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        # now produce the label and the image
        label = self.labels[idx]
        if self.preload_imgs:
            image = cheepre.provide_crop_img(self.largeimgs[self.img_list[self.img_idxs[idx]]], self.posis[idx], self.posjs[idx])
        else:
            image = cheepre.provide_crop(self.img_dir, self.img_list[self.img_idxs[idx]], self.posis[idx], self.posjs[idx])
        
        # transform
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
        
        #if self.add_dim:
        #    image = image.unsqueeze(0)
        # REMOVED: image already had size (1, 29, 29)
            
        return image.to(self.targetdevice), label.to(self.targetdevice)



