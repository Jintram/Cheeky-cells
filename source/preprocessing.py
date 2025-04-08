


import os

import re

import glob

from PIL import Image

import napari

import numpy as np

import cv2
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_triangle, threshold_isodata, threshold_minimum, apply_hysteresis_threshold
from skimage.morphology import opening, disk, square, binary_opening

import matplotlib.pyplot as plt

# from source.preprocessing import *
# reload
# import importlib; import source.preprocessing; importlib.reload(source.preprocessing)

def get_magnification_fromfilename(filename):
    # can handle magnification form 10X-99X (assumes digit, digit, X pattern)
    # filename = '3x Cellen (20x vegroting).jpg'
    
    # use regexp to extract a substring from string filename that agrees with the pattern number+number+(x or X)
    match = re.search(r'[0-9][0-9][xX]', filename)
    
    if match:        
        return int(match.group()[:2])    
    return None
    
    
    
def resize_pictures_infolder(input_folder, output_folder=None, newX=20):
    # newX=20; input_folder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW/'
    # newX is new magnification that image will be resized to be consistent with
    
    if output_folder==None:
        output_folder = input_folder.rstrip('/') + '_resized/'  
    os.makedirs(output_folder, exist_ok=True)
    
    # use glob.glob to get all files in input_folder
    list_all_files = glob.glob(input_folder + '/*')
    
    # loop over the files
    for idx, filepath in enumerate(list_all_files):
        # idx=0; filepath=list_all_files[0]
        
        # get the filename
        filename = filepath.split('/')[-1]
        
        # get the magnification
        magnification = get_magnification_fromfilename(filename)
        
        if magnification:
            # read the image
            img = Image.open(filepath)
            
            # calculate the new size
            new_size = (int(img.width * newX / magnification), int(img.height * newX / magnification))
            
            # resize the image
            img_resized = img.resize(new_size)
            
            # save the image to the output folder
            img_resized.save(output_folder + f'/resizedTo{newX}_' + filename)  
        else:
            print('No magnification found for file: ' + filename)
            
def wipe_borders(seg_layer_data, SIZE_HALF=14):
    
    # wipe the borders clean
    seg_layer_data[:SIZE_HALF+1, :] = 0 
    seg_layer_data[:, :SIZE_HALF+1] = 0 
    seg_layer_data[-SIZE_HALF-1:, :] = 0
    seg_layer_data[:, -SIZE_HALF-1:] = 0 
    
    return seg_layer_data
    
def convert_imgs_togrey(input_folder):
    
    # loop over all images in the folder, and convert them
    # to greyscale, export these to a 2nd folder with
    # suffix "_grey"
    
    # input_folder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW/'
    output_folder = input_folder.rstrip('/') + '_grey/'
    os.makedirs(output_folder, exist_ok=True)
    
    list_all_files = glob.glob(input_folder + '/*')
    
    for idx, filepath in enumerate(list_all_files):
        # idx = 0; filepath = list_all_files[0]
        
        img      = Image.open(filepath)
        img_grey = img.convert('L')  # Convert to greyscale
        
        # plt.imshow(img_grey); plt.show(); plt.close()
        
        # save the image to the output directory
        filename = filepath.split('/')[-1]
        img_grey.save(output_folder + filename)  
    

def annotate_pictures(input_folder, output_folder=None):
    # input_folder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized/'
    
    global continue_loop
    continue_loop = True
    
    # Output folder
    if output_folder==None:
        output_folder = input_folder.rstrip('/') + '_humanannotated/'
    os.makedirs(output_folder, exist_ok=True)
    
    # Get input images
    list_all_files = glob.glob(input_folder + '/*')
    
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
    
def give_img_paths(input_folder):
    
    # Get input images
    list_allimgpaths      = [os.path.basename(f) for f in glob.glob(input_folder + '/*')]
    # Get annotation files
    # list_annotfilepaths   = [re.sub(r'\..*$', '_seg.npy', f) for f in list_allimgpaths]
    list_annotfilepaths   = [os.path.splitext(f)[0]+'_seg.npy' for f in list_allimgpaths]
    
    return list_allimgpaths, list_annotfilepaths

    
def acquire_trainingset_info(input_folder, annot_dir=None):
    # input_folder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized/'
    '''
    Simply counts amount of annotated pixels in each img file.
    '''
    
    # now loop over the pixels in each annotated file, get
    # the annotated pixels (i.e. desired output labels),
    # and attach the correct training picture to it

    # Output folder
    if annot_dir==None:
        annot_dir = input_folder.rstrip('/') + '_humanannotated/'
    
    list_allimgpaths, list_annotfilepaths = give_img_paths(input_folder)
        
    # Loop over images in Napari, saving the annotated images
    # Stop the loop when SHIFT+Q is pressed in Napari
    
    annot_pixelcount_list = np.zeros(len(list_allimgpaths), dtype=np.uint64)
    for idx, (filename_img, filename_annot) in enumerate(zip(list_allimgpaths, list_annotfilepaths)):
        # idx=0; filename_img = list_allimgpaths[0]; filename_annot = list_annotfilepaths[0]
        
        # check whether annot file exists
        if not os.path.exists(annot_dir + filename_annot):
            print('No annotation file found for file: ' + filename_annot)
            continue
        
        # load image
        # img        = Image.open(input_folder + filename_img)
        img_annot  = np.load(annot_dir + filename_annot, allow_pickle=True)
            # plt.imshow(img_annot); plt.show(); plt.close()
        
        print(np.sum(img_annot>0),'pixels in image \''+ filename_annot+'\'')
        annot_pixelcount_list[idx] = np.sum(img_annot>0)
        
    return annot_pixelcount_list, list_allimgpaths, list_annotfilepaths
        
        
def build_labels_and_positions(annot_dir, annot_pixelcount_list, list_allimgpaths, list_annotfilepaths):
    '''
    Go over the annotated images, and save for each annotated pixel
    the corresponding image index, it's location, and its 
    annotation in a table.
    
    Such that:
    complete_label_table[0,:] = img_idx
    complete_label_table[1,:] = posi
    complete_label_table[2,:] = posj
    complete_label_table[3,:] = annotation
    '''
    
    total_labels = np.sum(annot_pixelcount_list)    
    idx_related_imgs = np.concatenate((np.array([0], dtype=np.uint64), np.cumsum(annot_pixelcount_list)))
    complete_label_table = np.empty([4, total_labels], dtype=np.uint64)
    for img_idx, (px_count, filename_img, filename_annot) in enumerate(zip(annot_pixelcount_list, list_allimgpaths, list_annotfilepaths)):
        # idx=0; filename_img = list_allimgpaths[0]; filename_annot = list_annotfilepaths[0]
        
        # check whether annotated pixels (implies existence file)
        if not px_count>0:
            continue
        
        img_annot  = np.load(annot_dir + filename_annot, allow_pickle=True)
        
        sel_annotated = img_annot>0
        loc_annotated = np.where(sel_annotated)
        annotations   = img_annot[sel_annotated]
        
        # now store
        idx1=idx_related_imgs[img_idx]; idx2=idx_related_imgs[img_idx+1]
        complete_label_table[0,idx1:idx2] = img_idx
        complete_label_table[1,idx1:idx2] = loc_annotated[0]
        complete_label_table[2,idx1:idx2] = loc_annotated[1]
        complete_label_table[3,idx1:idx2] = annotations        
        
        if complete_label_table.nbytes/(1024**2)>250: # size in megabytes
            Warning('The indexing table got pretty large, perhaps re-code this part..')
        
    return complete_label_table
        
def provide_crop(input_folder, filename_img, posi, posj, SIZE_HALF=14):
    # input_folder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/'
    
    # IMG_IDX = 100
    # filename_img = list_allimgpaths[complete_label_table[0,IMG_IDX]]
    # posi = complete_label_table[1, IMG_IDX]
    # posj = complete_label_table[2, IMG_IDX]
    
    # open image
    img = np.array(Image.open(input_folder + filename_img))
        
    # determine crop region
    cropregion = [posi-SIZE_HALF, posi+SIZE_HALF+1, posj-SIZE_HALF, posj+SIZE_HALF+1]    
    
    # show crop region on large image
    # plt.imshow(img); plt.gca().add_patch(plt.Rectangle((cropregion[2], cropregion[0]), cropregion[3]-cropregion[2], cropregion[1]-cropregion[0], fill=False, edgecolor='red', linewidth=2))
    # plt.show(); plt.close()
        
    img_crop = img[cropregion[0]:cropregion[1], cropregion[2]:cropregion[3]]
        #plt.imshow(img_crop); plt.show(); plt.close()
    
    return img_crop
    
def test_provide_crop(img_idx):    
    
    # img_idx=100
    
    input_folder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_grey/'
    annot_dir = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized_humanannotated/'
    
    annot_pixelcount_list, list_allimgpaths, list_annotfilepaths = acquire_trainingset_info(input_folder, annot_dir)
    complete_label_table = build_labels_and_positions(annot_dir, annot_pixelcount_list, list_allimgpaths, list_annotfilepaths)
    
    img_crop = provide_crop(input_folder=input_folder, 
                            filename_img = list_allimgpaths[complete_label_table[0,img_idx]],
                            posi=complete_label_table[1,img_idx], 
                            posj=complete_label_table[2,img_idx])    
    
    return(img_crop)

def test_test_provide_crop():
    
    IMG_IDX = 0
    plt.imshow(test_provide_crop(IMG_IDX))
    plt.show(); plt.close()
    
    IMG_IDX = 100
    plt.imshow(test_provide_crop(IMG_IDX))
    plt.show(); plt.close()
    
                     
    
#idx=0
#list_allimgpaths[idx]
#posi
#provide_crop(img_dir, img_list[idx], self.posis, self.posjs)    
        
def map_index_to_img(idx, annot_pixelcount_list):
    '''
    Currently not used.
    '''
    # annot_pixelcount_list=test_case
    
    if idx > np.nansum(annot_pixelcount_list):
        raise ValueError('idx not available')
    
    # find the first value equal or higher than idx in annot_pixelcount_list
    img_idx = np.where(np.nancumsum(annot_pixelcount_list) >= idx)[0][0]
    
    return img_idx

def test_map_index_to_img():    
    
    test_case = [10,20,10, np.nan, 30]
    
    assert map_index_to_img(0, test_case) == 0, 'failure'
    
    assert map_index_to_img(10, test_case) == 0, 'failure'
    assert map_index_to_img(11, test_case) == 1, 'failure'
    
    assert map_index_to_img(30, test_case) == 1, 'failure'
    assert map_index_to_img(31, test_case) == 2, 'failure'
    
    assert map_index_to_img(40, test_case) == 2, 'failure'
    assert map_index_to_img(70, test_case) == 4, 'failure'
    # map_index_to_img(71, test_case) # should fail
    
# def create_crop_img_n_label(input_folder, list_allimgpaths, annot_dir, list_annotfilepaths, img_idx):
    
#     filename_img   = list_allimgpaths[img_idx]
#     filename_annot = list_annotfilepaths[img_idx]
    
#     # Load the two images
#     img        = Image.open(input_folder + filename_img)
#     img_annot  = np.load(annot_dir + filename_annot, allow_pickle=True)
    
def divide_evenly(total_samples, count_list):
    # count_list = annot_pixelcount_list    
    
    pass
    
    # distribution = np.array([np.nan]*len(count_list))
    # distribution[np.logical_or(count_list==0, np.isnan(count_list))] = 0
    
    # slots_w_space = 
    
    # # determine how much will be added
    # while slots_w_space>0:
        
    #     to_add = 
        
    #     np.nanmin(count_list)
        
    #     slots_w_space = 
    
        
    
def create_sampled_imgset(total_samples):
    
    pass
    
    # distribute N samples (total_samples) evenly over
    # sample numbers lisetd in annot_pixelcount_list
    
    # annot_pixelcount_list, list_allimgpaths, list_annotfilepaths = acquire_trainingset_info(input_folder)
    
    
        
    