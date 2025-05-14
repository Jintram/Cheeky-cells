


!!!!!  !!!!!  !!!!!  !!!!!  !!!!!  !!!!!  !!!!!  !!!!!  
CONTINUE WORKING ON "applymodel_test"
!!!!!  !!!!!  !!!!!  !!!!!  !!!!!  !!!!!  !!!!!  !!!!!  

################################################################################

# =====
# General libraries
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor, Lambda
from torch.optim.lr_scheduler import StepLR, LambdaLR

import torch
import time   

import glob

# =====
# Fluoppi related 
import sys; sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/2024_Fluoppi/')
import source_2.fluoppi2_readwrite as flan42_rw
# import source.fluoppi_mainlib_v4 as flan4
# import source.fluoppi_config as flaconf
# import source.fluoppi_morestats as mstats
# from source.convenient_definitions import *

# =====
# Custom pytorch related libraries
import importlib
# import mypytorch_fullseg.mytrainer_fullseg as mt #; importlib.reload(mt)
# import mypytorch_fullseg.dataset_classes_fullseg as md # ; importlib.reload(md)
import mypytorch_fullseg.models_downloaded.unet_model as um #; importlib.reload(um)
import source.annotation_aided as aa

################################################################################

def load_sample_data_for_test():
    
    METADATA_FILE = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_Olaf-20250425.xlsx'
    
    # OUTPUTDIRANALYSIS = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_20250425_Olaf/'
    # import os; os.makedirs(OUTPUTDIRANALYSIS, exist_ok=True)
    
    # Now load the metadata
    import pandas as pd
    metadata_table = pd.read_excel(METADATA_FILE, header=0)
    
    # Now load the first image
    img_fluop, img_cells = flan42_rw.get_my_image_split(metadata_table, 0)
    
    # Show
    plt.imshow(np.log(.1+img_cells[2000:4000, 2000:4000])); plt.show(); plt.close()
    
    return img_cells

def split_image_to_tiles(img_cells, tilesize=500):
    # tilesize=500
    
    # now split the img_cells in an array with tilesize x tilesize (500 x 500) tiles
    row_splits = np.array_split(img_cells, img_cells.shape[0] // tilesize, axis=0)
    tiles = [np.array_split(row, row.shape[1] // tilesize, axis=1) for row in row_splits]
    tiles = [tile for row in tiles for tile in row]
    
    return tiles
    
def loadmodel():
    '''
    Load the model. The model was trained and stored in the file 
    mypytorch_fullseg/testing_code_fullseg.py
    '''
    
    modelUNet = um.UNet(n_channels=1, n_classes=4).to("mps")
    
    MODEL_WEIGHT_PATH = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/UNET_MODELS/modelUNet20250426_0929_LONGTRAINGOOD.pth'
    
    # now set the weights
    modelUNet.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
    
    
    return modelUNet
    

def applymodel_test():
    
    # Load sample data
    img_cells = load_sample_data_for_test()
    # Load the model
    modelUNet = loadmodel()
    
    # split the test image into tiles
    img_cells_tiled = split_image_to_tiles(img_cells)
    
    # rescale the tiles
    img_cells_tiled_rescaled = [aa.image_autorescale(tile) for tile in img_cells_tiled]
    
    # show a tile
    # Raw
    plt.imshow(img_cells_tiled[20]); plt.show(); plt.close()
    # Rescaled
    plt.imshow(img_cells_tiled_rescaled[20]); plt.show(); plt.close()