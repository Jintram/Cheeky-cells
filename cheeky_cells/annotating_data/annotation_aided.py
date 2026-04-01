
# %% #################################################################################

import os
import napari
import numpy as np
from PIL import Image
import glob

import nd2
import skimage as sk

from scipy.ndimage import maximum_filter
from skimage.filters import threshold_triangle, threshold_otsu, threshold_li

from skimage.morphology import remove_small_objects, label

import time

import matplotlib.pyplot as plt

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

       

def image_greyscale(img):
    '''
    Converts image to grey scale if the image has 3 dimensions.
    This is for auto-thresholding, which doesn't work well with 
    multiple channels.
    Greyscale images are not used as input for the the U-net.
    '''
    
    if len(img.shape)>2:
        img = img.mean(axis=2)
            
    return img
    
    
def basicseg1(img):
    '''
    very basic segmentation function
    '''
    thresholdval = threshold_otsu(img)-1
    img_segmaskauto = img > thresholdval
    thresholdval2 = threshold_otsu(img[img_segmaskauto])
    img_segmaskauto = img > thresholdval2
    return img_segmaskauto

def basicseg2(img):
    '''
    very basic segmentation function (version 2)
    '''
    # img = current_tile_rescaled
    
    thresholdval = threshold_li(img)
    
    img_segmaskauto = img>thresholdval
    
    # now remove small parts
    img_segmaskauto = remove_small_objects(img_segmaskauto, min_size=50**2)
    
    # close with circular kernel of radius 10
    selem = sk.morphology.disk(10)
    img_segmaskauto = sk.morphology.binary_closing(img_segmaskauto, selem)
    
        # plt.imshow(img_segmaskauto*1.0); plt.show()
        
    return img_segmaskauto
    

def annothelp_tile_and_segment(img_toseg, img_annot=None, TILE_SIZE=2000, segfn=basicseg1,
                               showedgepic=False, tile_selection_by='maxvar', showplots=True,
                               folder_devplots=None, rescalelog=True, bg_percentile=.2,
                               rescalegrey=True):
    '''    
    TO DO: This function is still slightly messy!
        Perhaps make it more modular?
    
    Goal is to produce a single tile from a larger image that can be used
    for annotation and training purposes later.
    - If image size < tile size, it pads the image.
    - If >1 tile, it chooses one by a criterium, such as max variance. 
        (or max signal 'maxsignal', or max high intensity area 'maxarea3bg')
        
    If a segmentation is already given (img_annot), it will also select a corresponding
    region from that segmentation mask. (This is especifically for image sets were
    some other segmentation algorithm, e.g. classical image operations based,
    was already performed, but didn't work well.)
    
    If no segmentation is already given, this function will also try to produce some 
    basic segmentation mask for the selected tile.
    
    It returns
        - img_toseg_tile; the original selected tile
        - img_seg0_tile; a basic segmentation mask based on otsu thresholding
        - img_toseg_tile_edges; the locally normalized tile, which enhances edges.
        - img_toseg_tile_rescaled; a log-rescaled version of the original tile
    '''
    
    # showedgepic=False; img_annot=None; tile_selection_by='maxvar'; showplots=True
    # TILE_SIZE=1000
    
    # PADDING THAT WAS DONE BEFORE IS UNNECESSARAY
    # img_toseg = paddimg_totilesize(img_toseg, TILE_SIZE)
    #     # plt.imshow(img_toseg); plt.show(); 
    # # also pad annotation if given
    # if not img_annot is None:
    #     img_annot = paddimg_totilesize(img_annot, TILE_SIZE)        
    #     # plt.imshow(img_annot); plt.show(); 
    
    # Just need to make the right split
    
    # now split the img_toseg in an array with approx. 2000x2000 tiles (or TILE_SIZE x TILE_SIZE)
    tile_nr_rows = np.max([1, img_toseg.shape[0] // TILE_SIZE])
    row_splits = np.array_split(img_toseg, tile_nr_rows, axis=0)
        # test1=np.array_split(testarray4500, testarray4500.shape[0] // TILE_SIZE, axis=0)
    tiles = [np.array_split(row, np.max([1, row.shape[1] // TILE_SIZE]), axis=1) for row in row_splits]
    tiles = [tile for row in tiles for tile in row]
        # plt.imshow(tiles[0]); plt.show(); plt.close()
        # plt.hist(tiles[0].ravel()); plt.show()
    
    # apply max filter with size 250    
    tiles_maxfilt = [maximum_filter(tile, size=20) for tile in tiles] # 250 or 20?
    # normalize tiles by neighborhood maxima, this will enhance edges
    tiles_normlocal = [tile/tile_mf for tile,tile_mf in zip(tiles, tiles_maxfilt)] 
        # plt.imshow(tiles_normlocal[0]); plt.show(); plt.close()
    
    # identify tile that we'll select, either by max variance or signal
    def getnnonzero(anarray):
        anarray[anarray==0] = np.min(anarray)
        return anarray    
    if tile_selection_by == 'maxvar':
        print('Selecting slice with max variance')
        idx_tile_sel = np.argmax([np.var(getnnonzero(tile)) for tile in tiles_normlocal])
    elif tile_selection_by == 'maxsignal':
        print('Selecting slice with max signal')
        idx_tile_sel   = np.argmax([np.sum(getnnonzero(tile)) for tile in tiles])
        # plt.imshow(tiles_normlocal[idx_tile_sel]); plt.show(); plt.close()
    elif tile_selection_by == 'maxarea3bg':
        print('Selecting slice with max area > 3*background (determined by percentile)')
        idx_tile_sel = np.argmax([np.sum(getnnonzero(tile)>np.percentile(getnnonzero(tile), .02)*3) for tile in tiles])
    else:
        print('Invalid option!')
        return None
        
    # now also select the corresponding segmentation region 
    if not img_annot is None:
        tiles_annot = [np.array_split(row, np.max([1, row.shape[1] // TILE_SIZE]), axis=1) for row in 
                    np.array_split(img_annot, np.max([1, img_annot.shape[0] // TILE_SIZE]), axis=0)]
        tiles_annot = [tile for row in tiles_annot for tile in row]
    
        # plot side by side
        if showplots:
            fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
            _=ax.imshow(tiles_normlocal[idx_tile_sel])
            _=ax.contour(tiles_annot[idx_tile_sel], levels=[0.5], colors='red', linewidths=0.5)
            plt.tight_layout()
            
            if not folder_devplots is None:
                plt.savefig(os.path.join(folder_devplots, 'tile_and_annotation.pdf'), dpi=300)
            else:
                plt.show(); plt.close()
                
    # generate a new segmentation of the tile based on otsu method
    #current_tile_log = subtractbaseline(np.log(tiles[idx_tile_sel] + .1))
    # current_tile_rescaled = image_autorescale(tiles[idx_tile_sel])
    current_tile_rescaled     = crw.image_autorescale(tiles[idx_tile_sel], rescalelog=rescalelog, bg_percentile=bg_percentile)
        # plt.imshow(tiles[idx_tile_sel]);plt.show()
        # plt.imshow(current_tile_rescaled); plt.show()
        
    # If no segmentation exists, just try something basic
    if img_annot is None:        
        current_tile_rescaledgrey = image_greyscale(current_tile_rescaled)
            # plt.imshow(current_tile_rescaledgrey); plt.show()
            
        # apply custom seg function
        if rescalegrey:
            img_segmaskauto = segfn(current_tile_rescaledgrey)
        else:
            img_segmaskauto = segfn(tiles[idx_tile_sel])
        
        # remove small objects 
        img_seg0_tile = remove_small_objects(img_segmaskauto, min_size=20**2)
        
        if showplots:
            fig, ax = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))
            ax[0].imshow(current_tile_rescaled)
            # ax[0].contour(img_segmaskauto, levels=[0.5], colors='red')
            #ax[1].imshow(tiles[idx_tile_sel])
            #ax[1].contour(img_segmaskauto, levels=[0.5], colors='white')
            ax[1].imshow(img_segmaskauto)
            plt.tight_layout()
            if not folder_devplots is None:
                plt.savefig(os.path.join(folder_devplots, 'tile_and_autoannotation.pdf'), dpi=300)
            else:
                plt.show(); plt.close()
    else:
        img_seg0_tile = tiles_annot[idx_tile_sel]
            
    if showedgepic:
        # edge image from the local normalization
        fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
        ax.imshow(tiles_normlocal[idx_tile_sel])
        ax.contour(img_segmaskauto, levels=[0.5], colors='red')    
        plt.show(); plt.close()
    
    # pic of cells
    img_toseg_tile = tiles[idx_tile_sel]
    img_toseg_tile_rescaled = current_tile_rescaled
    img_toseg_tile_edges = tiles_normlocal[idx_tile_sel]
    
    return img_toseg_tile, img_seg0_tile, img_toseg_tile_edges, img_toseg_tile_rescaled


# %% #################################################################################


def edit_annotation_napari(image, segmentation, mylabelcolormap=None, title='Edit seg file'):
    '''
    Opens napari with `image` as background and `segmentation` as an editable label
    layer, optionally styled with `mylabelcolormap`. Returns `(seg_data, quitloop_flag)`,
    where `seg_data` is the edited annotation (or None if quit) and `quitloop_flag`
    is True if the user pressed 'q'.
    '''
    quitloop_flag = False

    viewer = napari.Viewer(title=title)
    viewer.add_image(image, name='Original image')
    seg_layer = viewer.add_labels(name='Annotation',
                                  data=segmentation,
                                  colormap=mylabelcolormap)

    def _on_quit(event=None):
        nonlocal quitloop_flag
        quitloop_flag = True
        viewer.close()

    viewer.bind_key('q', _on_quit)
    napari.run()

    if quitloop_flag:
        return None, True

    return seg_layer.data, False


def annotate_pictures_aided(df_metadata, file_idx, 
                            output_segfolder, 
                            segfn = basicseg1,
                            intitial_segfolder = None, TILE_SIZE=2000,
                            tile_selection_by = 'maxvar', 
                            ignore_saved_file=False, showplots=False, rescalelog=True, bg_percentile=.2, showrawimg=False,
                            rescalegrey=True, mylabelcolormap=None,
                            mynaparifunction=None,
                            file_infix='_tile'):
    '''
    
    '''
    # df_metadata should exist
    # file_idx = 0
    #
    # intitial_segfolder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheek-Cells_resized_anonymous_seg25/'
    # output_segfolder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/humanseg/'
    #
    # tile_selection_by = 'maxvar'; ignore_saved_file=False; showplots=False; TILE_SIZE=1000; rescalelog=False; segfn=basicseg2
    
    if file_idx is None:
        raise ValueError('file_idx should be specified.')
    
    # Make segdir 
    os.makedirs(output_segfolder, exist_ok=True)
    
    # Get image name
    _, _, filename, _, _ = crw.get_fileinfo_metadata(df_metadata, file_idx)
    filename_base = os.path.splitext(filename)[0]
    print(f'Load file: {filename}')
        
    # Load the image
    img_toseg = \
        crw.loadimgfile_metadata(df_metadata, file_idx)
        # plt.imshow(img_toseg); plt.show()

    # Load seg file if present
    if not intitial_segfolder is None:
        img_annot = crw.loadsegfile_metadata(df_metadata, file_idx, intitial_segfolder)
    else:
        img_annot = None
        # plt.imshow(img_annot); plt.show()
        
    # Now get a single tile for the annotation
    img_toseg_tile, img_seg0_tile, img_toseg_tile_edges, img_toseg_tile_rescaled = \
        annothelp_tile_and_segment(
            img_toseg, 
            img_annot=img_annot, TILE_SIZE=TILE_SIZE, segfn=segfn,
            showedgepic=False, 
            tile_selection_by=tile_selection_by, showplots=showplots,
            folder_devplots=None, rescalelog=rescalelog, bg_percentile=bg_percentile,
            rescalegrey=rescalegrey)
        # plt.imshow(img_toseg_tile); plt.show()
        # plt.imshow(img_seg0_tile); plt.show()
        # plt.imshow(img_toseg_tile_rescaled); plt.show()
        # plt.imshow(img_toseg_tile_edges); plt.show()
        
    # Prepare output filenames
    filename_base = os.path.splitext(filename)[0]
    newfilename_annot        = filename_base + file_infix + '_seg.npy'
    newfilename_img          = filename_base + file_infix + '_img.npy' 
    newfilename_img_enhanced = filename_base + file_infix + '_img_enhanced.npy'
    newfilename_extra        = filename_base + file_infix + '_transform.npy' # edge transform
        
    # if seg file already exists, load it (assumes also other files match)
    # also check for .npz (phase 3 output format)
    newfilename_annot_npz = newfilename_annot.replace('.npy', '.npz')
    if os.path.exists(output_segfolder + newfilename_annot) and (not ignore_saved_file):
        print('Loaded image seg file:', newfilename_annot)
        img_seg0_tile = np.load(output_segfolder + newfilename_annot, allow_pickle=True)
    elif os.path.exists(output_segfolder + newfilename_annot_npz) and (not ignore_saved_file):
        print('Loaded image seg file (.npz):', newfilename_annot_npz)
        npz_data = np.load(output_segfolder + newfilename_annot_npz, allow_pickle=True)
        img_seg0_tile = npz_data['img_pred_lbls']
            # plt.imshow(img_seg0_tile); plt.show(); plt.close()
            
    # now improve this image in napari 
    display_image = img_toseg_tile if showrawimg else img_toseg_tile_rescaled
    str_windowtitle = 'Editing file ' + filename_base
    # call the napari edit function
    if mynaparifunction is None:
        mynaparifunction = edit_annotation_napari  
    seg_layer_data, quitloop_flag = mynaparifunction(display_image, img_seg0_tile, mylabelcolormap, title=str_windowtitle)
        
    if quitloop_flag:
        return quitloop_flag
        
    # now save the annotation data
    print('Saving annotation data')
    np.save(output_segfolder + newfilename_annot, seg_layer_data)
    np.save(output_segfolder + newfilename_img, img_toseg_tile)
    np.save(output_segfolder + newfilename_extra, img_toseg_tile_edges)
    np.save(output_segfolder + newfilename_img_enhanced, img_toseg_tile_rescaled)
    
    # plot the result
    if showplots:
        fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
        ax.imshow(img_toseg_tile)
        ax.contour(seg_layer_data, levels=[0.5], colors='red')    
        plt.show(); plt.close()
    
    return quitloop_flag

# %%

# Now a function that loops over the images that are available in the metadata and calls
# annotate_pictures_aided()
def annotate_all_pictures_aided(df_metadata, output_segfolder, 
                                intitial_segfolder = None, TILE_SIZE=2000,
                                tile_selection_by = 'maxvar', segfn=basicseg1,
                                ignore_saved_file=False, showplots=False,
                                rescalelog = True, bg_percentile=.2, showrawimg=False,
                                rescalegrey = True, mylabelcolormap = None,
                                mynaparifunction = None,
                                file_infix = '_tile'):
    '''
    
    '''
    # df_metadata should exist
    # file_idx = 0
    #
    # intitial_segfolder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheek-Cells_resized_anonymous_seg25/'
    # output_segfolder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/humanseg/'
    #
    # tile_selection_by = 'maxvar'; ignore_saved_file=False; showplots=False; TILE_SIZE=1000
    
    if df_metadata is None:
        raise ValueError('df_metadata should be specified.')
    
    # Make segdir 
    os.makedirs(output_segfolder, exist_ok=True)
    
    # Loop over all files in the metadata
    for file_idx in range(df_metadata.shape[0]): # file_idx = 0
    # for file_idx in range(11, df_metadata.shape[0]): 
        print('=======')
        print(f'Processing file {file_idx+1} of {df_metadata.shape[0]}')
        quitloop_flag = \
            annotate_pictures_aided(df_metadata, file_idx, 
                                output_segfolder, 
                                segfn=segfn,
                                intitial_segfolder = intitial_segfolder, 
                                TILE_SIZE=TILE_SIZE,
                                tile_selection_by = tile_selection_by, 
                                ignore_saved_file=ignore_saved_file, 
                                showplots=showplots,
                                rescalelog=rescalelog, 
                                bg_percentile=bg_percentile, 
                                showrawimg=showrawimg,
                                rescalegrey=rescalegrey,
                                mylabelcolormap=mylabelcolormap,
                                mynaparifunction=mynaparifunction,
                                file_infix=file_infix)
        if quitloop_flag:
            print('Quitting annotation loop as requested by user.')
            break
        
                            
                            
                            
        # read key, "n" for next, or "q" for quit
        # user_input = input('Press "n" for next image, or "q" to quit: ').strip().lower()
        # if user_input == 'q':
        #     print('Quitting annotation loop.')
        #     break


# %% ################################################################################
# Discarded code


# PADDING IS UNNECESSARAY
# def paddimg_totilesize(img, TILE_SIZE=2000):
#     '''
#     Extend image to desired square dimensions, if necessary
    
#     Pad with zeroes as these are probably easily recognized by U-net 
#     '''
    
#     # if the image size is below tile size, pad image to tile size
#     if np.any(np.array(img.shape[:2])<TILE_SIZE):
        
#         # Padding size for 2d
#         pad_width = [(0, max(0, TILE_SIZE - img.shape[0])),
#                      (0, max(0, TILE_SIZE - img.shape[1]))]
        
#         # Add padding size for 3rd dimension if necessary
#         if len(img.shape) == 3: pad_width.extend([(0, 0)])
        
#         # Pad and return the result
#         return np.pad(img, pad_width, mode='constant', constant_values=0)

