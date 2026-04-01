
# %% ################################################################################
# libraries

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from skimage.measure import label

cm_to_inch = 1/2.54

import cheeky_cells.readwrite.cheeky_readwrite as crw
    # import importlib; importlib.reload(crw)
import cheeky_cells.annotating_data.annotation_aided as caa

# %% ################################################################################


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
    annotation_feature_labeled = np.zeros_like(initial_annotation_human)
    # notice that order is important, img_proximityzones overlaps w/ original, needs to go first
    annotation_feature_labeled[img_proximityzones>0] = 3
    annotation_feature_labeled[initial_annotation_human>0] = 1
    annotation_feature_labeled[img_boundaries>0] = 2    
        # plt.imshow(annotation_feature_labeled); plt.show(); plt.close()
    
    end_time = int(time.time())
    print('Function took ', end_time-start_time, 'seconds to execute.')
    
    return annotation_feature_labeled
    
# %% ################################################################################    

    

def postprocess_basicsegfile(df_metadata, file_idx, segfolder, suffix, showplot=False, updateFN=update_seg_extrannot):
    
    # Load a test segmentation mask
    file_index = 0
    current_mask = \
        crw.loadsegfile_metadata(df_metadata, file_idx, segfolder, silence=False, suffix='_tile_seg')
    
    # Update the mask
    current_mask_updated = updateFN(current_mask)
    
    # Now save the mask
    crw.savesegfile_default(current_mask_updated, df_metadata, file_idx, segfolder, suffix=suffix)
    
    
    # now show
    if showplot:
        plt.imshow(current_mask); plt.show()
        plt.imshow(current_mask_updated); plt.show()


def postprocess_basicsegfile_all(df_metadata, segfolder, suffix='_tile_seg_postpr', showplot=False, updateFN=update_seg_extrannot):
    '''
    Post-process all files in metadata such that the training
    classes will be improved.
    Basic seg files will be converted according to 'updateFN'.
    
    Example: convert simple binary regions to labeled mask
    with inside region, borders, proximity zones and background.
    '''
    
    # loop over all files
    for file_idx in range(len(df_metadata)):
        print('Processing file index', file_idx, 'of', len(df_metadata))
        postprocess_basicsegfile(df_metadata, file_idx, segfolder, suffix=suffix, showplot=showplot, updateFN=updateFN)





# %% ################################################################################    
# PROBABLY OBSOLETE    
    
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
    
    thefilelist_imgs, thefilelist_annot, thefilelist_extra, \
        thefilelist_enhanced, thefilelist_annot_features = \
                                get_file_list_annotimgs(folderconfig)
    
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
        current_annot          = np.load(annothuman_folder + thefilelist_annot[FILE_IDX], allow_pickle=True)
        
        # now get improved annotation
        print('Processing sample', thefilelist_annot[FILE_IDX],'(', FILE_IDX, ')')
        current_annot_withfeatures = update_seg_extrannot(current_annot)
        
        # now save the result
        np.save(annothuman_folder + thefilelist_annot[FILE_IDX].replace('.npy', '_features.npy'), current_annot_withfeatures)

        