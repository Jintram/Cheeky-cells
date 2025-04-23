
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

from skimage.morphology import remove_small_objects, label
from skimage.exposure import rescale_intensity

import time

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
    #current_tile_log = subtractbaseline(np.log(tiles[idx_tile_sel] + .1))
    current_tile_rescaled = image_autorescale(tiles[idx_tile_sel])
    thresholdval = threshold_otsu(current_tile_rescaled)-1
    img_segmaskauto = current_tile_rescaled > thresholdval
    thresholdval2 = threshold_otsu(current_tile_rescaled[img_segmaskauto])
    img_segmaskauto = current_tile_rescaled > thresholdval2
    
    if showplots:
        fig, ax = plt.subplots(1, 2, figsize=(5*cm_to_inch, 10*cm_to_inch))
        ax[0].imshow(current_tile_rescaled)
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
    img_cells_tile_rescaled = current_tile_rescaled
    img_cells_tile_edges = tiles_norm[idx_tile_sel]
    
    return img_cells_tile, img_seg0_tile, img_cells_tile_edges, img_cells_tile_rescaled


################################################################################

def annotate_pictures_aided(FILE_IDX_user=None, folderconfig=None, tile_selection_by = 'maxvar', ignore_saved_file=False, showplots=False):
    '''
    This function was never called, but executed line by line manually.
    
    NOTE:
    image_autorescale is now applied only before the image is shown to the user, 
    but it might be smart to also use that input image for thresholding segmentation.
    '''
    # tile_selection_by = 'maxvar'; ignore_saved_file=False; showplots=False
    
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
    FILE_IDX=0 # This should be made input of this function
    if not FILE_IDX_user == None: 
        FILE_IDX = FILE_IDX_user
        
    # Load data
    img_fluop, img_cells = flrw.get_my_image_split(metadata_table, FILE_IDX)
    
    # Show current annotation if desired
    if False:
        show_current_annot(initial_seg_folder, list_all_annotfiles, img_cells)
    
    # Get high-var ±2000px tile and 
    # print('TO DO: image_autorescale is not applied before auto-segment yet!') # done with img_cells_tile_rescaled
    img_cells_tile, img_seg0_tile, img_cells_tile_edges, img_cells_tile_rescaled = \
        annothelp_tile_and_segment(img_cells, showedgepic=False, img_annot=None, tile_selection_by=tile_selection_by, showplots=showplots)
    
    # prepare output file names
    newfilename_annot = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_annothuman.npy')
    newfilename_img   = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_img.npy')
    newfilename_img_enhanced   = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_img_enhanced.npy')
    newfilename_extra = list_all_annotfiles[FILE_IDX].replace('_seg.npy','_tile_transform.npy')
    
    # if seg file already exists, load it (assumes also other files match)
    if os.path.exists(output_seg_folder + newfilename_annot) and (not ignore_saved_file):
        print('Loaded image seg file:', newfilename_annot)
        img_seg0_tile = np.load(output_seg_folder + newfilename_annot, allow_pickle=True)
            # plt.imshow(img_seg0_tile); plt.show(); plt.close()

    # now improve this image in napari
    viewer = napari.Viewer()
    # img_cells_tile_rescaled = image_autorescale(img_cells_tile)
    viewer.add_image(img_cells_tile_rescaled)

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
    np.save(output_seg_folder + newfilename_img_enhanced, img_cells_tile_rescaled)
    
    # plot the result
    if showplots:
        fig, ax = plt.subplots(1, 1, figsize=(5*cm_to_inch, 5*cm_to_inch))
        ax.imshow(img_cells_tile)
        ax.contour(seg_layer_data, levels=[0.5], colors='red')    
        plt.show(); plt.close()
    


# THE ANNOTATION FUNCTION
def perform_seg():    
    
    folderconfig = {
        'input_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/20250328_FLUOPPI/',
        'initial_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_202504_V2files_Exp-20250328/seg_20250313_135502/segmentation/',
        'output_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/',
        'metadata_file': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_data20250328.xlsx'}
            
    # methods: {maxarea3bg, maxvar, maxsignal}
    
    annotate_pictures_aided(FILE_IDX_user=0, folderconfig=folderconfig) # REVISED
    annotate_pictures_aided(FILE_IDX_user=1, folderconfig=folderconfig) # REVISED
    annotate_pictures_aided(FILE_IDX_user=2, folderconfig=folderconfig) # REVISED
    annotate_pictures_aided(FILE_IDX_user=3, folderconfig=folderconfig) # REVISED
    annotate_pictures_aided(FILE_IDX_user=4, folderconfig=folderconfig) # REVISED
                
    annotate_pictures_aided(FILE_IDX_user=5, folderconfig=folderconfig) # REVISED
    annotate_pictures_aided(FILE_IDX_user=6, folderconfig=folderconfig) # REVISED
    annotate_pictures_aided(FILE_IDX_user=7, folderconfig=folderconfig) # REVISED
    annotate_pictures_aided(FILE_IDX_user=8, folderconfig=folderconfig) # REVISED
    
    # annotate_pictures_aided(FILE_IDX_user=9, folderconfig=folderconfig, tile_selection_by='maxsignal', ignore_saved_file=True)
    annotate_pictures_aided(FILE_IDX_user=9, folderconfig=folderconfig, tile_selection_by='maxarea3bg') # REVISED
    annotate_pictures_aided(FILE_IDX_user=10, folderconfig=folderconfig, tile_selection_by='maxarea3bg') # REVISED






def get_file_list_annotimgs(folderconfig):
    
    # Load metadata, get filenames
    metadata_table = pd.read_excel(folderconfig['metadata_file'])
    list_all_imgfiles_original = metadata_table['filename'].values
    
    # Now get the tiles
    thefilelist_imgs =  [X.replace('.nd2', '_tile_img.npy') for X in list_all_imgfiles_original]
    thefilelist_annot = [X.replace('.nd2', '_tile_annothuman.npy') for X in list_all_imgfiles_original]
    thefilelist_extra = [X.replace('.nd2', '_tile_transform.npy') for X in list_all_imgfiles_original]
    thefilelist_enhanced = [X.replace('.nd2', '_tile_img_enhanced.npy') for X in list_all_imgfiles_original]
    
    return thefilelist_imgs, thefilelist_annot, thefilelist_extra, thefilelist_enhanced
         

def update_seg_extrannot(initial_annotation_human):
    '''
    Based on simply binary mask with non-touching cells, 
    create a segmentation with additional annotation:
    Output is labeled image with:
    0. Background
    1. Cell area
    2. Cell boundary
    3. Proximity zones, where cells (almost) touch
    
    NOTE
    This function is extremely slow due to the many dilation operations.
    This could be sped up by isolating cells into bboxes, expand, and 
    update only the corresponding bbox in the target image. 
    I did this before. 
    '''
    # initial_annotation_human=current_annot
    
    print('Working on sample..')
    start_time = int(time.time())
    
    # first get labeled mask and the unique labels
    mask_labeled = label(initial_annotation_human)
    the_labels = np.unique(mask_labeled[mask_labeled>0])
    
    # loop over each region, expand it, and add it to a new image
    # overlapping regions will get values > 1
    img_labels_intermediate = np.zeros_like(initial_annotation_human)
    disk10 = sk.morphology.disk(10)
    for current_label in the_labels:
        # current_label = the_labels[0]
        
        # mask for current region
        current_mask = mask_labeled == current_label        
        # expand by 10 pixels
        current_mask_exp = sk.morphology.binary_dilation(current_mask, disk10)        
        # add the mask to the image
        img_labels_intermediate += current_mask_exp.astype(np.uint8)
        
        if current_label % 10 == 0:
            print('Number of processed cells:', current_label)
    
    # determine zones were cells almost touch, proximity zones
    img_proximityzones = img_labels_intermediate>1    
        # plt.imshow(img_labels_intermediate); plt.show(); plt.close()    
        # plt.imshow(img_touchingzones); plt.show(); plt.close()
    
    # Now identify the cell edges; 
    # This is simpler: subtract eroded from original
    disk5 = sk.morphology.disk(5)
    img_boundaries = initial_annotation_human - sk.morphology.binary_erosion(initial_annotation_human, disk5)
        # plt.imshow(img_boundaries); plt.show(); plt.close()
    
    # Now create the final image, using the identified cell edges and proximity zones
    annotation_feature_labeled = np.copy(initial_annotation_human)
    annotation_feature_labeled[img_boundaries>0] = 2
    annotation_feature_labeled[img_proximityzones>0] = 3    
        # plt.imshow(annotation_feature_labeled); plt.show(); plt.close()
    
    end_time = int(time.time())
    print('Function took ', end_time-start_time, 'seconds to execute.')
    
    return annotation_feature_labeled
    
def post_processing():
    
    folderconfig = {
        'input_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/20250328_FLUOPPI/',
        'initial_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_202504_V2files_Exp-20250328/seg_20250313_135502/segmentation/',
        'output_seg_folder': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/',
        'metadata_file': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_data20250328.xlsx',
        'plotdir': '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/_plots/'}
    
    annothuman_folder = folderconfig['output_seg_folder']
    plotdir = folderconfig['plotdir']
    os.makedirs(plotdir, exist_ok=True)
       
    # thepath = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
    
    thefilelist_imgs, thefilelist_annot, thefilelist_extra, thefilelist_enhanced = get_file_list_annotimgs(folderconfig)
    
    # load the first one
    # WRONG: 5, 6, 7, 8, 
    for FILE_IDX in range(11):
        # FILE_IDX = 0
    
        current_img   = np.load(annothuman_folder + thefilelist_imgs[FILE_IDX], allow_pickle=True)
        current_annot = np.load(annothuman_folder + thefilelist_annot[FILE_IDX], allow_pickle=True)
        
        new_img = image_autorescale(current_img)
        
        fig, ax = plt.subplots(1, 1, figsize=(20*cm_to_inch, 20*cm_to_inch))
        ax.imshow(new_img)
        ax.contour(current_annot, levels=[0.5], colors='red')    
        plt.tight_layout()
        plt.savefig(plotdir + thefilelist_imgs[FILE_IDX].replace('.npy', '_groundtruthhuman.pdf'), dpi=300)
        plt.close(fig)
        
    for FILE_IDX in range(11):
        # FILE_IDX = 0
    
        current_img_enhanced   = np.load(annothuman_folder + thefilelist_enhanced[FILE_IDX], allow_pickle=True)
        current_annot = np.load(annothuman_folder + thefilelist_annot[FILE_IDX], allow_pickle=True)
        
        # now get improved annotation
        print('Processing sample', thefilelist_annot[FILE_IDX])
        current_annot_withfeatures = update_seg_extrannot(current_annot)
        
        # now save the result
        np.save(annothuman_folder + thefilelist_annot[FILE_IDX].replace('.npy', '_features.npy'), current_annot_withfeatures)

        
    