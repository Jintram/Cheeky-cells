
#################################################################################

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

#################################################################################

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


def annothelp_tile_and_segment(img_cells, showedgepic=False, img_annot=None, tile_selection_by='maxvar', showplots=True):
    
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
        
    
    # now also select the correspdoning segmentation region 
    if not img_annot == None:
        tiles_annot = [np.array_split(row, row.shape[1] // 2000, axis=1) for row in 
                    np.array_split(img_annot, img_annot.shape[0] // 2000, axis=0)]
        tiles_annot = [tile for row in tiles_annot for tile in row]
    
        # plot side by side
        if showplots:
            fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
            ax.imshow(tiles_norm[idx_tile_sel])
            ax.contour(tiles_annot[idx_tile_sel], levels=[0.5], colors='white')
            plt.show(); plt.close()
    
    # generate a new segmentation of the tile based on otsu method
    def subtractbaseline(anarray, est_base_pct=.2):
        thresholdlow = np.percentile(anarray, est_base_pct)
        anarray[anarray<=thresholdlow] = thresholdlow
        return anarray
    current_tile_log = subtractbaseline(np.log(tiles[idx_tile_sel] + .1))
    thresholdval = threshold_otsu(current_tile_log)-1
    img_segmaskauto = current_tile_log > thresholdval
    thresholdval2 = threshold_otsu(current_tile_log[img_segmaskauto])
    img_segmaskauto = current_tile_log > thresholdval2
    
    if showplots:
        fig, ax = plt.subplots(1, 2, figsize=(5*cm_to_inch, 10*cm_to_inch))
        ax[0].imshow(current_tile_log)
        ax[0].contour(img_segmaskauto, levels=[0.5], colors='red')
        ax[1].imshow(tiles[idx_tile_sel])
        ax[1].contour(img_segmaskauto, levels=[0.5], colors='white')
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
    img_cells_tile = tiles[idx_tile_sel]
    img_cells_tile_edges = tiles_norm[idx_tile_sel]
    
    return img_cells_tile, img_seg0_tile, img_cells_tile_edges


################################################################################

def annotate_pictures_aided(FILE_IDX_user=None, folderconfig=None, tile_selection_by = 'maxvar', ignore_saved_file=False, showplots=False):
    '''
    This function was never called, but executed line by line manually.
    '''
    
    if folderconfig==None:
        folderconfig = {
            'input_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/20250328_FLUOPPI/',
            'initial_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_202504_V2files_Exp-20250328/seg_20250313_135502/segmentation/',
            'output_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/',
            'metadata_file': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_data20250328.xlsx'}
        
    input_folder       = folderconfig['input_folder']
    initial_seg_folder = folderconfig['initial_seg_folder']
    output_seg_folder  = folderconfig['output_seg_folder']
    metadata_file      = folderconfig['metadata_file']
    
    os.makedirs(output_seg_folder, exist_ok=True)
        
    # Load metadata, get filenames
    metadata_table = pd.read_excel(metadata_file)
    list_all_imgfiles = metadata_table['filename'].values
    list_all_annotfiles = [P.replace('.nd2','_seg.npy') for P in list_all_imgfiles]
    
    # now load first image and annotation
    FILE_IDX=5 # This should be made input of this function
    if not FILE_IDX_user == None: 
        FILE_IDX = FILE_IDX_user
        
    # Load data
    img_fluop, img_cells = flrw.get_my_image_split(metadata_table, FILE_IDX)
    
    # Show current annotation if desired
    if False:
        show_current_annot(initial_seg_folder, list_all_annotfiles, img_cells)
    
    # Get high-var ±2000px tile and 
    img_cells_tile, img_seg0_tile, img_cells_tile_edges = \
        annothelp_tile_and_segment(img_cells, showedgepic=False, img_annot=None, tile_selection_by=tile_selection_by, showplots=showplots)
    
    # prepare output file names
    newfilename_annot = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_annothuman.npy')
    newfilename_img   = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_img.npy')
    newfilename_extra = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_transform.npy')
    
    # if seg file already exists, load it (assumes also other files match)
    if os.path.exists(output_seg_folder + newfilename_annot) and (not ignore_saved_file):
        print('Loaded image seg file:', newfilename_annot)
        img_seg0_tile = np.load(output_seg_folder + newfilename_annot, allow_pickle=True)
            # plt.imshow(img_seg0_tile); plt.show(); plt.close()

    # now improve this image in napari
    viewer = napari.Viewer()
    viewer.add_image(img_cells_tile)

    # add a label layer                           
    seg_layer = viewer.add_labels(name='segmentation', data=img_seg0_tile)
    # viewer.close()
    napari.run()
        
    # now save the annotation data
    print('Saving annotation data')
    seg_layer_data = seg_layer.data
    np.save(output_seg_folder + newfilename_annot, seg_layer_data)
    np.save(output_seg_folder + newfilename_img, img_cells_tile)
    np.save(output_seg_folder + newfilename_extra, img_cells_tile_edges)
    
    # plot the result
    if showplots:
        fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
        ax.imshow(img_cells_tile)
        ax.contour(seg_layer_data, levels=[0.5], colors='red')    
        plt.show(); plt.close()


def perform_seg():    
    
    folderconfig = {
        'input_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/20250328_FLUOPPI/',
        'initial_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_202504_V2files_Exp-20250328/seg_20250313_135502/segmentation/',
        'output_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/',
        'metadata_file': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_data20250328.xlsx'}
            
    # methods: {maxarea3bg, maxvar, maxsignal}
    
    annotate_pictures_aided(FILE_IDX_user=0, folderconfig=folderconfig)
    annotate_pictures_aided(FILE_IDX_user=1, folderconfig=folderconfig)
    annotate_pictures_aided(FILE_IDX_user=2, folderconfig=folderconfig)
    annotate_pictures_aided(FILE_IDX_user=3, folderconfig=folderconfig)
    annotate_pictures_aided(FILE_IDX_user=4, folderconfig=folderconfig)
    
            
    annotate_pictures_aided(FILE_IDX_user=5, folderconfig=folderconfig) # FIXED
    annotate_pictures_aided(FILE_IDX_user=6, folderconfig=folderconfig) # REDO!  
    annotate_pictures_aided(FILE_IDX_user=7, folderconfig=folderconfig) # REDO!
    annotate_pictures_aided(FILE_IDX_user=8, folderconfig=folderconfig) # REDO!
    
    # annotate_pictures_aided(FILE_IDX_user=9, folderconfig=folderconfig, tile_selection_by='maxsignal', ignore_saved_file=True)
    annotate_pictures_aided(FILE_IDX_user=9, folderconfig=folderconfig, tile_selection_by='maxarea3bg')    
    annotate_pictures_aided(FILE_IDX_user=10, folderconfig=folderconfig, tile_selection_by='maxarea3bg')

def get_file_list_annotimgs(folderconfig):
    
    # Load metadata, get filenames
    metadata_table = pd.read_excel(folderconfig['metadata_file'])
    list_all_imgfiles_original = metadata_table['filename'].values
    
    # Now get the tiles
    
    thefilelist_imgs =  [X.replace('.nd2', '_tile_img.npy') for X in list_all_imgfiles_original]
    thefilelist_annot = [X.replace('.nd2', '_tile_annothuman.npy') for X in list_all_imgfiles_original]
    thefilelist_extra = [X.replace('.nd2', '_tile_transform.npy') for X in list_all_imgfiles_original]
    
    return thefilelist_imgs, thefilelist_annot, thefilelist_extra
    

def post_processing():
    
    folderconfig = {
        'input_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/20250328_FLUOPPI/',
        'initial_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_202504_V2files_Exp-20250328/seg_20250313_135502/segmentation/',
        'output_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/',
        'metadata_file': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_data20250328.xlsx'}
    annothuman_folder = folderconfig['output_seg_folder']
       
    # thepath = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
    
    thefilelist_imgs, thefilelist_annot, thefilelist_extra = get_file_list_annotimgs(folderconfig)
    
    # load the first one
    # WRONG: 5, 6, 7, 8, 
    FILE_IDX = 5
    
    current_img   = np.load(annothuman_folder + thefilelist_imgs[FILE_IDX], allow_pickle=True)
    current_annot = np.load(annothuman_folder + thefilelist_annot[FILE_IDX], allow_pickle=True)
    
    plt.imshow(current_img)
    plt.contour(current_annot, levels=[0.5], colors='red')    
    plt.show(); plt.close()
    