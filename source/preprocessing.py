
import os

import re

import glob

from PIL import Image



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