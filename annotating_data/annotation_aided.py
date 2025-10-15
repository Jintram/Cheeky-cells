
# %% #################################################################################

import os
import napari
import numpy as np
from PIL import Image
import glob

import nd2
import skimage as sk

from scipy.ndimage import maximum_filter
from skimage.filters import threshold_triangle, threshold_otsu

from skimage.morphology import remove_small_objects, label
from skimage.exposure import rescale_intensity

import time

import matplotlib.pyplot as plt

# import sys; sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/2024_Fluoppi/')
# import source_2.fluoppi2_readwrite as flrw

import pandas as pd

# %% #################################################################################
# More customized stuff

# folder_devplots= '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/devplots/'

import readwrite.cheeky_readwrite as crw
    # import importlib; importlib.reload(crw)

cm_to_inch = 1/2.54

# %% #################################################################################

# About the annotation
#
# - Segmentation files are outputted to folder "outputdirectory"
#  
# - User can provide files that already have some segmentation, these should be
#   binary mask files. Separate cells are separated by zeroes.
#
# 


# %% #################################################################################

def subtractbaseline(anarray, est_base_pct=.2):
    thresholdlow = np.percentile(anarray, est_base_pct)
    anarray[anarray<=thresholdlow] = thresholdlow
    return anarray


def image_autorescale(input_img):
    
    new_img = np.log(.1+subtractbaseline(input_img))
    # rescale to range 0-255
    new_img = rescale_intensity(new_img, 'image', (0, 255))
    new_img = new_img.astype(np.uint8)
    
    return new_img
       

def paddimg_totilesize(img, TILE_SIZE=2000):
    '''
    extend image to desired square dimensions, if necessary
    '''
    
    # if the image size is below tile size, pad image to tile size
    if np.any(np.array(img.shape[:2])<TILE_SIZE):
        
        # Padding size for 2d
        pad_width = [(0, max(0, TILE_SIZE - img.shape[0])),
                     (0, max(0, TILE_SIZE - img.shape[1]))]
        
        # Add padding for 3rd dimension if necessary
        if len(img.shape) == 3: pad_width.extend([(0, 0)])
        
        # Return the result
        return np.pad(img, pad_width, mode='constant', constant_values=0)


def do_rescale_togrey(img):
    '''
    Converts image to grey scale if the image has 3 dimensions.
    This is for auto-thresholding, which doesn't work well with 
    multiple channels.
    '''
    
    if len(img.shape)>2:
        img = img.mean(axis=2)
        
    # rescale
    img = rescale_intensity(img, in_range='image', out_range=(0,255))        
    
    return img
    

def annothelp_tile_and_segment(img_toseg, img_annot=None, TILE_SIZE=2000,
                               showedgepic=False, tile_selection_by='maxvar', showplots=True,
                               folder_devplots=None):
    
    # showedgepic=False; img_annot=None; tile_selection_by='maxvar'; showplots=True
    # TILE_SIZE=1000
    
    # pad image if necessary
    img_toseg = paddimg_totilesize(img_toseg, TILE_SIZE)
        # plt.imshow(img_toseg); plt.show(); 
    # also pad annotation if given
    if not img_annot is None:
        img_annot = paddimg_totilesize(img_annot, TILE_SIZE)
        
        # plt.imshow(img_annot); plt.show(); 
    
    # now split the img_toseg in an array with 2000x2000 tiles
    row_splits = np.array_split(img_toseg, img_toseg.shape[0] // TILE_SIZE, axis=0)
    tiles = [np.array_split(row, row.shape[1] // TILE_SIZE, axis=1) for row in row_splits]
    tiles = [tile for row in tiles for tile in row]
        # plt.imshow(tiles[0]); plt.show(); plt.close()
    
    # apply max filter with size 250    
    tiles_maxfilt = [maximum_filter(tile, size=20) for tile in tiles] # 250 or 20?
    # normalize tiles
    tiles_norm = [tile/tile_mf for tile,tile_mf in zip(tiles, tiles_maxfilt)] 
        # plt.imshow(tiles_norm[0]); plt.show(); plt.close()
    
    # identify tile that we'll select, either by max variance or signal
    def getnnonzero(anarray):
        anarray[anarray==0] = np.min(anarray)
        return anarray    
    if tile_selection_by == 'maxvar':
        print('Selecting slice with max variance')
        idx_tile_sel = np.argmax([np.var(getnnonzero(tile)) for tile in tiles_norm])
    elif tile_selection_by == 'maxsignal':
        print('Selecting slice with max signal')
        idx_tile_sel   = np.argmax([np.sum(getnnonzero(tile)) for tile in tiles])
        # plt.imshow(tiles_norm[idx_tile_sel]); plt.show(); plt.close()
    elif tile_selection_by == 'maxarea3bg':
        print('Selecting slice with max area > 3*background (determined by percentile)')
        idx_tile_sel = np.argmax([np.sum(getnnonzero(tile)>np.percentile(getnnonzero(tile), .02)*3) for tile in tiles])
    else:
        print('Invalid option!')
        return None
        
    
    # now also select the corresponding segmentation region 
    if not img_annot is None:
        tiles_annot = [np.array_split(row, row.shape[1] // TILE_SIZE, axis=1) for row in 
                    np.array_split(img_annot, img_annot.shape[0] // TILE_SIZE, axis=0)]
        tiles_annot = [tile for row in tiles_annot for tile in row]
    
        # plot side by side
        if showplots:
            fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
            _=ax.imshow(tiles_norm[idx_tile_sel])
            _=ax.contour(tiles_annot[idx_tile_sel], levels=[0.5], colors='red', linewidths=0.5)
            plt.tight_layout()
            
            if not folder_devplots is None:
                plt.savefig(os.path.join(folder_devplots, 'tile_and_annotation.pdf'), dpi=300)
            else:
                plt.show(); plt.close()
            
    
    # generate a new segmentation of the tile based on otsu method
    #current_tile_log = subtractbaseline(np.log(tiles[idx_tile_sel] + .1))
    # current_tile_rescaled = image_autorescale(tiles[idx_tile_sel])
    current_tile_rescaled = do_rescale_togrey(tiles[idx_tile_sel])
    thresholdval = threshold_otsu(current_tile_rescaled)-1
    img_segmaskauto = current_tile_rescaled > thresholdval
    thresholdval2 = threshold_otsu(current_tile_rescaled[img_segmaskauto])
    img_segmaskauto = current_tile_rescaled > thresholdval2
    
    if showplots:
        fig, ax = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))
        ax[0].imshow(current_tile_rescaled)
        # ax[0].contour(img_segmaskauto, levels=[0.5], colors='red')
        ax[1].imshow(tiles[idx_tile_sel])
        ax[1].contour(img_segmaskauto, levels=[0.5], colors='white')
        plt.tight_layout()
        if not folder_devplots is None:
            plt.savefig(os.path.join(folder_devplots, 'tile_and_autoannotation.pdf'), dpi=300)
        else:
            plt.show(); plt.close()
        
    
    if showedgepic:
        # edge image from the local normalization
        fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
        ax.imshow(tiles_norm[idx_tile_sel])
        ax.contour(img_segmaskauto, levels=[0.5], colors='red')    
        plt.show(); plt.close()
    
    # remove small objects 
    img_seg0_tile = remove_small_objects(img_segmaskauto, min_size=20**2)
    
    # pic of cells
    img_toseg_tile = tiles[idx_tile_sel]
    img_toseg_tile_rescaled = current_tile_rescaled
    img_toseg_tile_edges = tiles_norm[idx_tile_sel]
    
    return img_toseg_tile, img_seg0_tile, img_toseg_tile_edges, img_toseg_tile_rescaled


# %% #################################################################################

def get_segfile_ifany(df_metadata, file_idx, intitial_segfolder):
    '''
    Looks for existing seg file in the form of _seg.npy or _seg.tif based
    on df_metadata, file_idx.
    '''
    
    # Get image info
    _, _, filename, _, _ = crw.get_fileinfo_metadata(df_metadata, file_idx)
    
    # Get current file extension
    filename_extension = os.path.splitext(filename)[1]
    
    # Check for existing seg file
    filepath_npy = os.path.join(intitial_segfolder, filename.replace(filename_extension, '_seg.npy'))
    filepath_tif = os.path.join(intitial_segfolder, filename.replace(filename_extension, '_seg.tif'))
    
    # Load the file
    if os.path.exists(filepath_npy):
        img_annot = np.load(filepath_npy)
    elif os.path.exists(filepath_tif):
        img_annot = sk.io.imread(filepath_tif)
        
    # Flatten to 2d
    if len(img_annot.shape) == 3:            
        img_annot = img_annot.max(axis=2)
        
    # Return it
    return img_annot

def annotate_pictures_aided(df_metadata, file_idx, 
                            output_segfolder, intitial_segfolder = None, 
                            folderconfig=None, tile_selection_by = 'maxvar', ignore_saved_file=False, showplots=False):
    '''
    
    '''
    # df_metadata should exist
    # file_idx = 0
    #
    # intitial_segfolder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheek-Cells_resized_anonymous_seg25/'
    # output_segfolder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/humanseg/'
    #
    # tile_selection_by = 'maxvar'; ignore_saved_file=False; showplots=False
    
    if file_idx is None:
        raise ValueError('file_idx should be specified.')
    
        
    # Load the image
    img_toseg = \
        crw.loadsegfile_metadata(df_metadata, file_idx)

    # Load seg file if present
    img_annot = get_segfile_ifany(df_metadata, file_idx, intitial_segfolder)
    
    
    
    fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))
    axs[0].imshow(img_toseg)
    if not img_annot is None:
        axs[1].imshow(img_annot)
    plt.show()


