


import os
import napari
import numpy as np
from PIL import Image
import glob

import nd2
import skimage as sk

from scipy.ndimage import maximum_filter
from skimage.filters import threshold_triangle, threshold_otsu

from skimage.morphology import remove_small_objects

import matplotlib.pyplot as plt

import sys; sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/2024_Fluoppi/')
import source_2.fluoppi2_readwrite as flrw

import pandas as pd

cm_to_inch = 1/2.54



def show_current_annot(initial_seg_folder, list_all_annotfiles, img_cells):
    '''Get the current annotation and show both'''
    img_annot = np.load(initial_seg_folder + list_all_annotfiles[FILE_IDX], allow_pickle=True)
    
    if not img_cells.shape == img_annot.shape:
        print('Image and annotation shape do not match!')
        return
        
    # show the image and the annotation
    fig, ax = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))
    ax[0].imshow(img_cells[0:2000, 0:2000], cmap='viridis')
    ax[0].contour(img_annot[0:2000, 0:2000] > 0, colors='red', levels=[0.5])
    ax[1].imshow(img_annot[0:2000, 0:2000], cmap='gray')    
    plt.tight_layout()
    plt.show()

def annothelp_tile_and_segment(img_cells, showedgepic=False, img_annot=None):
    
    # now split the img_cells in an array with 2000x2000 tiles
    row_splits = np.array_split(img_cells, img_cells.shape[0] // 2000, axis=0)
    tiles = [np.array_split(row, row.shape[1] // 2000, axis=1) for row in row_splits]
    tiles = [tile for row in tiles for tile in row]
        # plt.imshow(tiles[0]); plt.show(); plt.close()
    
    # apply max filter with size 250    
    tiles_maxfilt = [maximum_filter(tile, size=20) for tile in tiles] # 250 or 20?
    # normalize tiles
    tiles_norm = [tile/tile_mf for tile,tile_mf in zip(tiles, tiles_maxfilt)] 
        # plt.imshow(tiles_norm[0]); plt.show(); plt.close()
    
    # identify tile with most variance in normalized thingy
    def getnnonzero(anarray):
        anarray[anarray==0] = np.min(anarray)
        return anarray
    max_variance_idx = np.argmax([np.var(getnnonzero(tile)) for tile in tiles_norm])
        # plt.imshow(tiles_norm[max_variance_idx]); plt.show(); plt.close()

    # now also select the correspdoning segmentation region 
    if not img_annot == None:
        tiles_annot = [np.array_split(row, row.shape[1] // 2000, axis=1) for row in 
                    np.array_split(img_annot, img_annot.shape[0] // 2000, axis=0)]
        tiles_annot = [tile for row in tiles_annot for tile in row]
    
        # plot side by side
        fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
        ax.imshow(tiles_norm[max_variance_idx])
        ax.contour(tiles_annot[max_variance_idx], levels=[0.5], colors='white')
        plt.show(); plt.close()

    # generate a new segmentation of the tile based on otsu method
    def subtractbaseline(anarray, est_base_pct=.2):
        thresholdlow = np.percentile(anarray, est_base_pct)
        anarray[anarray<=thresholdlow] = thresholdlow
        return anarray
    current_tile_log = subtractbaseline(np.log(tiles[max_variance_idx] + .1))
    thresholdval = threshold_otsu(current_tile_log)-1
    img_segmaskauto = current_tile_log > thresholdval
    thresholdval2 = threshold_otsu(current_tile_log[img_segmaskauto])
    img_segmaskauto = current_tile_log > thresholdval2
    fig, ax = plt.subplots(1, 2, figsize=(5*cm_to_inch, 10*cm_to_inch))
    ax[0].imshow(current_tile_log)
    ax[0].contour(img_segmaskauto, levels=[0.5], colors='red')
    ax[1].imshow(tiles[max_variance_idx])
    ax[1].contour(img_segmaskauto, levels=[0.5], colors='white')
    plt.show(); plt.close()
    
    if showedgepic:
        # edge image from the local normalization
        fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
        ax.imshow(tiles_norm[max_variance_idx])
        ax.contour(img_segmaskauto, levels=[0.5], colors='red')    
        plt.show(); plt.close()

    # remove small objects 
    img_seg0_tile = remove_small_objects(img_segmaskauto, min_size=20**2)
    
    # pic of cells
    img_cells_tile = tiles[max_variance_idx]
    
    return img_cells_tile, img_seg0_tile


def annotate_pictures_aided(input_folder, initial_seg_folder, output_folder=None):
    '''
    This function was never called, but executed line by line manually.
    '''
    
    input_folder = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/20250328_FLUOPPI/'
    initial_seg_folder = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_202504_V2files_Exp-20250328/seg_20250313_135502/segmentation/'
    output_seg_folder = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
    metadata_file = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_data20250328.xlsx'
    
    os.makedirs(output_seg_folder, exist_ok=True)
    
    global continue_loop
    continue_loop = True
    
    # Load metadata, get filenames
    metadata_table = pd.read_excel(metadata_file)
    list_all_imgfiles = metadata_table['filename'].values
    list_all_annotfiles = [P.replace('.nd2','_seg.npy') for P in list_all_imgfiles]
    
    # # Get input images
    # list_all_annotfiles = glob.glob(initial_seg_folder + '/*')
    # # Now convert to file list
    # filenames_annot = [P.split('/')[-1] for P in list_all_annotfiles]
    # # Also get basefilenames
    # basefilenames_annot = [P.replace('_seg.npy','') for P in filenames_annot]
    
    # now load first image and annotation
    FILE_IDX=5
    img_fluop, img_cells = flrw.get_my_image_split(metadata_table, FILE_IDX)
    
    # Show current annotation if desired
    if False:
        show_current_annot(initial_seg_folder, list_all_annotfiles, img_cells)
    
    # Get high-var ±2000px tile and 
    img_cells_tile, img_seg0_tile = annothelp_tile_and_segment(img_cells, showedgepic=False, img_annot=None)
    

    # to continue with an image:
    outputfilename = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_annothuman.npy')
    if os.path.exists(output_seg_folder + outputfilename):
        img_seg0 = np.load(output_seg_folder + outputfilename, allow_pickle=True)

    
    
    # now improve this image in napari
    viewer = napari.Viewer()
    viewer.add_image(tiles[max_variance_idx])

    # add a label layer                           
    seg_layer = viewer.add_labels(name='segmentation', data=img_seg0)
    # viewer.close()
    
    # now save the annotation data
    seg_layer_data = seg_layer.data
    newfilename_annot = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_annothuman.npy')
    newfilename_img   = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_img.npy')
    newfilename_extra = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_transform.npy')
    np.save(output_seg_folder + newfilename_annot, seg_layer_data)
    np.save(output_seg_folder + newfilename_img, tiles[max_variance_idx])
    np.save(output_seg_folder + newfilename_extra, tiles_norm[max_variance_idx])
    
    # plot the result
    fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
    ax.imshow(tiles[max_variance_idx])
    ax.contour(seg_layer_data, levels=[0.5], colors='red')    
    plt.show(); plt.close()
    
    #####
    # BELOW PART WASN'T USED YET..
    
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
    