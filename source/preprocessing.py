


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
        
        img = Image.open(filepath)
        img = np.array(img)
        
        viewer = napari.Viewer()
        viewer.add_image(img)
        
        # Load/initialize current seg
        if os.path.exists(output_folder + filename + '_seg.npy'):
            current_seg = np.load(output_folder + filename + '_seg.npy', allow_pickle=True)
        else:
            current_seg = np.zeros(np.shape(img)[:2]
        
        # add a label layer                           
        seg_layer = viewer.add_labels(name='segmentation', data=current_seg, dtype=np.uint8))
        
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
        
        napari.run()
        
        # close napari
        # viewer.close()
        
        # save the polygon layer to a numpy file
        np.save(output_folder + filename + '_seg.npy', seg_layer.data)
        
        if not continue_loop:
            break
    
    
    
            
            
def annotate_pictures_REMOVE(input_folder, output_folder=None):
    # input_folder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized/'
    ''''
    Loop over the pictures in the folder, open them with Napari,
    create an annotation layer, allow user to create polygons,
    save the layer to a numpy file with similar filename
    and _seg.
    '''
    
    if output_folder==None:
        output_folder = input_folder.rstrip('/') + '_humanannotated/'
    os.makedirs(output_folder, exist_ok=True)
    
    # again, loop over the pictures in the folder
    list_all_files = glob.glob(input_folder + '/*')
    
    for idx, filepath in enumerate(list_all_files):
        # idx=0; filepath=list_all_files[0]
        filename = filepath.split('/')[-1]
        
        img = Image.open(filepath)
        img = np.array(img)
        
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # invert image scale
        img_greyscale = 255 - img_greyscale

        # determine otsu threshold on img_greyscale
        thresholdval        = threshold_triangle(img_greyscale)
        img_segmaskauto     = img_greyscale > thresholdval
        # img_segmask_o  = binary_opening(img_segmask_o, disk(10))
        # use cv2 to perform binary opening
        img_segmaskauto = cv2.morphologyEx(img_segmaskauto.astype(np.uint8), cv2.MORPH_CLOSE, disk(20))
        # apply morphological opening from skimage library
        
        
        # plt.imshow(img_segmask_o); plt.show(); plt.close()
        
        viewer = napari.Viewer()
        viewer.add_image(img_greyscale)
        # add a label layer
        #seg_layer = viewer.add_labels(name='segmentation', data=np.zeros(np.shape(img)[:2], dtype=np.uint8))
        seg_layer = viewer.add_labels(name='segmentationauto', data=img_segmaskauto)

        # viewer.close()
        
        
        seg_mask = viewer.layers['segmentation'].data

        # let's threshold at 1% of the cell values
        threshold    = np.percentile(img_greyscale[seg_mask==1], 0.98)
        img_cellmask = np.zeros_like(img_greyscale)
        img_cellmask[img_greyscale<threshold] = 1
        

        
        
        
        plt.hist(img_greyscale[seg_mask==1].flatten(), bins=100, label='background')
        plt.hist(img_greyscale[seg_mask==2].flatten(), bins=100, label='foreground')
        plt.axvline(threshold)
        plt.legend()
        plt.show(); plt.close()


        
        viewer = napari.Viewer()
        viewer.add_image(img)
        # add a label layer
        seg_layer = viewer.add_labels(name='mask_thresholded', data=img_cellmask)
        

        seg_layer.data
        viewer.layers['empty_labels'].data

        # retrieve the labeled layer
        plt.imshow(img_cellmask); plt.show(); plt.close()
        
        napari.run()
        
        # close napari
        viewer.close()
        
        # save the polygon layer to a numpy file
        np.save(output_folder + filename + '_seg.npy', polygon_layer.data)
        
        # create an image of equal size as img
        img_seg = np.zeros_like(img)
        # now draw the polygons from the polygon_layer on top
        for polygon in polygon_layer.data:
            # polygon is a list of points
            # we need to convert it to a list of tuples
            polygon = [(int(point[0]), int(point[1])) for point in polygon]
            # now draw the polygon on img_seg
            img_seg = cv2.fillPoly(img_seg, [np.array(polygon)], 1)