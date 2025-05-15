


################################################################################

# =====
# General libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib.colors import ListedColormap

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor, Lambda
from torch.optim.lr_scheduler import StepLR, LambdaLR

from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation

import torch
import time   
import math

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

def load_sample_data_for_test(showit=False):
    
    METADATA_FILE = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_Olaf-20250425.xlsx'
    
    # OUTPUTDIRANALYSIS = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_20250425_Olaf/'
    # import os; os.makedirs(OUTPUTDIRANALYSIS, exist_ok=True)
    
    # Now load the metadata
    metadata_table = pd.read_excel(METADATA_FILE, header=0)
    
    # Now load the first image
    img_fluop, img_cells = flan42_rw.get_my_image_split(metadata_table, 0)
    
    # Show
    if showit:
        plt.imshow(np.log(.1+img_cells[2000:4000, 2000:4000])); plt.show(); plt.close()
    
    return img_cells

def split_image_to_tiles(img_cells, tilesize=500):
    # tilesize=500
    
    # expand the dimensions of the image to a multiple of 500
    img_cells_padded = np.pad(img_cells, ((0, tilesize - img_cells.shape[0] % tilesize), (0, tilesize - img_cells.shape[1] % tilesize)), mode='constant')
    
    # now split the img_cells in an array with tilesize x tilesize (500 x 500) tiles
    tiles_shape = (img_cells_padded.shape[0] // tilesize, img_cells_padded.shape[1] // tilesize)
    row_splits = np.array_split(img_cells_padded, img_cells_padded.shape[0] // tilesize, axis=0)
    tiles = [np.array_split(row, row.shape[1] // tilesize, axis=1) for row in row_splits]
    tiles = [tile for row in tiles for tile in row]
    
    return tiles, tiles_shape
    
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


def make_prediction_for_img500(img_input, myMLmodel):
    '''
    Use the model myMLmodel to create a prediction for 
    an input image (for the modelUNet model, 
    a size of 500 x 500 is required).
    
    The model requires the image to be rescaled according 
    to source.annotation_aided.image_autorescale().
    '''
    
    # img_input = img_cells_rescaled_tiled[24]
    
    # Convert a tile to a tensor        
    img_input = np.expand_dims(img_input, axis=0)
    img_input_tensor = torch.tensor(img_input, dtype=torch.float32).unsqueeze(0).to("mps")
    
    # now feed the tensor to the model
    pred = myMLmodel(img_input_tensor)
    # and assign categories by max
    pred_class = pred.argmax(1).cpu().detach().numpy()
        
    return pred_class


def applymodel_test():
    
    # Load sample data
    img_cells = load_sample_data_for_test()
    # Load the model
    modelUNet = loadmodel()
    
    # Rescale the input image
    img_cells_rescaled = aa.image_autorescale(img_cells)
    
    # split the image into tiles
    img_cells_rescaled_tiled, tiles_shape = split_image_to_tiles(img_cells_rescaled)
    # For reference, also split the unscaled image into tiles
    img_cells_tiled, _ = split_image_to_tiles(img_cells)

    # show a tile
    # Raw
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_cells_tiled[24])
    axs[1].imshow(img_cells_rescaled_tiled[24])
    plt.show(); plt.close()
    
    if False:
        # test for one tile
        
        # make prediction
        pred_class = make_prediction_for_img500(img_cells_rescaled_tiled[24], modelUNet)

        # show original image and prediction side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_cells_rescaled_tiled[24])    
        ax[1].imshow(pred_class[0]); 
        plt.show(); plt.close()
        
    # Now apply this to all tiles, in batches of 32.
    batch_size = 8
    all_preds = []

    # Process each batch, converting to tensor within the loop to save memory
    current_time = time.time()
    print('Current time: '+str(time.strftime("%H:%M:%S", time.localtime())))
    total_batches=round(np.ceil(len(img_cells_rescaled_tiled)/batch_size))
    for batch_idx in range(0, len(img_cells_rescaled_tiled), batch_size):
        # batch_idx=0
        
        print(f"Currently hanlding batch {round(batch_idx/batch_size+1)} of {total_batches}")        
        
        # Select tiles and convert to appropriate dimensions
        batch_tiles = np.expand_dims(np.array(img_cells_rescaled_tiled[batch_idx:batch_idx+batch_size]), axis=1)
        
        # Convert to tensor
        batch_tiles_tensor = torch.tensor(batch_tiles, dtype=torch.float32, requires_grad=False).to("mps")
        
        # Make the prediction
        with torch.no_grad():
            preds = modelUNet(batch_tiles_tensor)
            preds_class = preds.argmax(1).cpu().numpy()
            all_preds.extend(preds_class)   
        
        print(f"{(time.time() - current_time)/60:.2f} minutes working")
        
    # Optionally, visualize one of the predictions            
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(batch_tiles[0][0])
    ax[1].imshow(all_preds[0])
    plt.show()
    plt.close()
    
    # now paste together the tiles, using tiles_shape
    # Reconstruct the full prediction image from the tiles
    rows, cols = tiles_shape
    tile_h, tile_w = all_preds[0].shape
    full_pred = np.zeros((rows * tile_h, cols * tile_w), dtype=all_preds[0].dtype)

    for idx, tile in enumerate(all_preds):
        row = idx // cols
        col = idx % cols
        full_pred[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w] = tile

    # Optionally, crop to original image size if needed
    full_pred_cropped = full_pred[:img_cells.shape[0], :img_cells.shape[1]]

    # Show the reconstructed prediction
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(aa.image_autorescale(img_cells[0:3000, 0:3000]))
    ax[1].imshow(full_pred_cropped[0:3000, 0:3000])
    plt.title("Reconstructed Prediction")
    plt.show()
    plt.close()
    
    # label the image for post-processing
    full_pred_cropped_labeled = label(full_pred_cropped)
    
    # create a randomized version of the 'jet' map
    # get the jet colors as array
    # shuffle that array
    # put them back into cmap format
    # create a randomized version of the 'jet' map

    # Get the jet colormap as an array of RGBA colors
    jet = cm.get_cmap('jet', np.max(full_pred_cropped_labeled)+1)
    jet_colors = jet(np.arange(jet.N))

    # Shuffle the colors, but keep the first color (background) unchanged
    rng = np.random.default_rng(seed=42)
    shuffled_colors = jet_colors.copy()
    if len(shuffled_colors) > 1:
        shuffled_colors[1:] = rng.permutation(jet_colors[1:])

    # Create a new colormap from the shuffled colors
    random_jet_cmap = ListedColormap(shuffled_colors)
    
    # display the labeled image in random colors
    plt.imshow(full_pred_cropped_labeled[0:3000, 0:3000], cmap=random_jet_cmap)
    plt.show()
    plt.close()
    
    # create a background mask and dilate it
    background_mask = (full_pred_cropped==0)
    background_mask_dilated = binary_dilation(background_mask)
    background_bound_mask = background_mask_dilated-background_mask
    plt.imshow(background_bound_mask[0:3000,0:3000], cmap='grey')#, vmin=0, vmax=255)
    plt.show(); plt.close()
    
    # now also generate an overlap between the expanded background and 
    # the cytoplasm parts
    cytoplasm_mask = full_pred_cropped==1
    cytoplasm_touching_background = np.logical_and(cytoplasm_mask, background_bound_mask)
    
    # now identify labels of cytosplasm that touches the backround
    cyto_idx_to_remove = np.unique(full_pred_cropped_labeled[cytoplasm_touching_background])
    full_pred_cropped_processed = full_pred_cropped.copy()
    full_pred_cropped_processed[np.isin( full_pred_cropped_labeled, cyto_idx_to_remove )] = 0
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(full_pred_cropped[0:3000, 0:3000])
    ax[0].imshow(cytoplasm_touching_background[0:3000, 0:3000], cmap='Reds', alpha=1.0*cytoplasm_touching_background[0:3000, 0:3000])
    ax[1].imshow(full_pred_cropped_processed[0:3000, 0:3000])
    plt.show(); plt.close()
    
    # such that we have the final result
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_cells_rescaled)
    ax.contour(full_pred_cropped_processed==1, levels=[.5], colors='white')
    plt.show(); plt.close()
    
    # remove areas below a certain size
    CELL_AREA_LOWER_BOUND = 10**2*math.pi
    
    # from full_pred_cropped_processed, remove areas below CELL_AREA_LOWER_BOUND
    labeled_cells = label(full_pred_cropped_processed == 1)
    cell_props = regionprops(labeled_cells)
    cellidxs_to_keep = [prop.label for prop in cell_props if prop.area > CELL_AREA_LOWER_BOUND]
    full_pred_cropped_processed = np.isin(labeled_cells, cellidxs_to_keep)
    
    