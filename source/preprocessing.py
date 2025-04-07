


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
        np.save(output_folder + annotfilepath, seg_layer.data)   
        if not continue_loop:
            break
    
    
def generate_trainingset(input_folder, input_folder_seg=None):
    
    
    # now loop over the pixels in each annotated file, get
    # the annotated pixels (i.e. desired output labels),
    # and attach the correct training picture to it

    # Output folder
    if annot_folder==None:
        annot_folder = input_folder.rstrip('/') + '_humanannotated/'
    
    # Get input images
    list_allimgpaths      = [os.path.basename(f) for f in glob.glob(input_folder + '/*')]
    # Get annotation files
    # list_annotfilepaths   = [re.sub(r'\..*$', '_seg.npy', f) for f in list_allimgpaths]
    list_annotfilepaths   = [os.path.splitext(f)[0]+'_seg.npy' for f in list_allimgpaths]
    
    # Loop over images in Napari, saving the annotated images
    # Stop the loop when SHIFT+Q is pressed in Napari
    
    for idx, (filename_img, filename_annot) in enumerate(zip(list_allimgpaths, list_annotfilepaths)):
        # idx=0; filename_img = list_allimgpaths[0]; filename_annot = list_annotfilepaths[0]
        
        # check whether annot file exists
        if not os.path.exists(annot_folder + filename_annot):
            print('No annotation file found for file: ' + filename_annot)
            continue
        
        # load image
        img        = Image.open(input_folder + filename_img)
        img_annot  = np.load(annot_folder + filename_annot, allow_pickle=True)
            # plt.imshow(img_annot); plt.show(); plt.close()
        
        print(np.sum(img_annot>0),'pixels to annotate')
        # Oops, this is a bit much .. 
        # I guess I'll have to write a generator that directly accesses these annotations and images