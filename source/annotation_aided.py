


import os
import napari
import numpy as np
from PIL import Image
import glob

import nd2
import skimage as sk

from scipy.ndimage import maximum_filter

import matplotlib.pyplot as plt

import sys; sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/2024_Fluoppi/')
import source_2.fluoppi2_readwrite as flrw

import pandas as pd

def annotate_pictures_aided(input_folder, initial_seg_folder, output_folder=None):
    '''
    input_folder = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/20250328_FLUOPPI/'
    initial_seg_folder = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_202504_V2files_Exp-20250328/seg_20250313_135502/segmentation/'
    output_seg_folder = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
    metadata_file = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_data20250328.xlsx'
    '''
    
    metadata_table = pd.read_excel(metadata_file)
    
    global continue_loop
    continue_loop = True
    
    # 
    list_all_imgfiles = metadata_table['filename'].values
    list_all_annotfiles = [P.replace('.nd2','_seg.npy') for P in list_all_imgfiles]
    
    # # Get input images
    # list_all_annotfiles = glob.glob(initial_seg_folder + '/*')
    # # Now convert to file list
    # filenames_annot = [P.split('/')[-1] for P in list_all_annotfiles]
    # # Also get basefilenames
    # basefilenames_annot = [P.replace('_seg.npy','') for P in filenames_annot]
    
    # now load first image and annotation
    FILE_IDX=0
    img_fluop, img_cells = flrw.get_my_image_split(metadata_table, FILE_IDX)
    img_annot = np.load(initial_seg_folder + filenames_annot[FILE_IDX], allow_pickle=True)
    
    # show the image and the annotation
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_cells)
    ax[1].imshow(img_annot)
    plt.show(); plt.close()
    
    # now split the img_cells in an array with 2000x2000 tiles
    row_splits = np.array_split(img_cells, img_cells.shape[0] // 2000, axis=0)
    tiles = [np.array_split(row, row.shape[1] // 2000, axis=1) for row in row_splits]
    tiles = [tile for row in tiles for tile in row]
    plt.imshow(tiles[0]); plt.show(); plt.close()
    
    # get one tile
    current_tile = tiles[0]
    # apply max filter with size 250    
    tiles_maxfilt = [maximum_filter(tile, size=250) for tile in tiles]
    # normalize tiles
    tiles_norm = [tile/tile_mf for tile,tile_mf in zip(tiles, tiles_maxfilt)]
        # plt.imshow(tiles_norm[0]); plt.show(); plt.close()
    
    
    
    
    # trying out something else
    tiles_maxfiltsmall = [maximum_filter(tile, size=20) for tile in tiles]
    tiles_norm_small = [tile/tile_mf for tile,tile_mf in zip(tiles, tiles_maxfiltsmall)]
    plt.imshow(tiles_norm_small[0]); plt.show(); plt.close()

    # apply sobel for edge detection on tile 0
    sobel_tile = sk.filters.sobel(tiles[0])
    plt.imshow(sobel_tile); plt.show(); plt.close()

    # resize tile 0 to 20x smaller using max
    tile_0_small = sk.transform.rescale(tiles[0], 0.2, order=0, anti_aliasing=False)
    # apply sobel
    sobel_tile_0_small = sk.filters.sobel(tile_0_small)
    plt.imshow(sobel_tile_0_small); plt.show(); plt.close()
    
        
    # now identify the tile with the most variance
    # for each of the tiles, apply a local max filter of size 250
    
    
    tiles_filtered = [sk.filters.rank.maximum(tile, sk.morphology.footprint_rectangle(250)) for tile in tiles]
    
    
    max_variance_idx = np.argmax([np.var(tile) for tile in tiles])
    plt.imshow(tiles[max_variance_idx]); plt.show(); plt.close()
    
    # Loop over images in Napari, saving the annotated images
    # Stop the loop when SHIFT+Q is pressed in Napari
    
    for idx, filepath in enumerate(list_all_files):
        # idx=0; filepath=list_all_files[0]
        filename = filepath.split('/')[-1]
        annotfilepath = os.path.splitext(filename)[0] + '_seg.npy'
        
        img = Image.open(filepath)
        img = np.array(img)
        
        viewer = napari.Viewer()
        viewer.add_image(img)
        
        # Load/initialize current seg
        if os.path.exists(output_folder + annotfilepath):
            current_seg = np.load(output_folder + annotfilepath, allow_pickle=True)
        else:
            current_seg = np.zeros(np.shape(img)[:2], dtype=np.uint8)
        
        # add a label layer                           
        seg_layer = viewer.add_labels(name='segmentation', data=current_seg)
                
        # add a key binding (shift+Q) that makes 
        @viewer.bind_key('Shift+Q', overwrite=True)
        def stop_loop(viewer):            
            global continue_loop
            continue_loop = False
            print("Loop stop requested - will exit after this image")
            viewer.close()
        
        @viewer.bind_key('Shift+N', overwrite=True)
        def next_image(viewer):
            print("Skipping to next image")
            viewer.close()
        
        # Can't do this if want to use from Jupyter
        napari.run()
        
        # close napari
        # viewer.close()
        
        # save the polygon layer to a numpy file
        seg_layer_data_cleanborder = wipe_borders(seg_layer.data)
        np.save(output_folder + annotfilepath, seg_layer_data_cleanborder)
        if not continue_loop:
            break
    