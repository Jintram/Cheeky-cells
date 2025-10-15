

# %% ################################################################################

# This file contains a few small functions that aid in loading images, using 
# metadata.


# %% ################################################################################


import os
import nd2
import numpy as np
import skimage.io as skio
import glob

import pandas as pd

# %% ################################################################################
# 

def gen_metadatafile(basedirectory, subdirs, outputdirectory, 
                     file_formats=['.tif','.nd2','.jpg'], segchannel = 'all', invert_image='no', 
                     default_columns = ['dataset', 'train_or_test', 'comments']):
                                        # 'basedir', 'subdir' will always be added    
    '''
    Given a folder, generate an excel file with column 'filename' with all image files
    from a given directory, and add also default columns that the user can 
    fill in further.
    
    If you have (multiple) subdirectories with images in your main data directory,
    specify those in subdirs.
    
    Use trailling slashes in paths.
    
    If segchannel = np.nan, it will later be assumed to take all channels.
    
    invert_image can be 'yes' or 'no', to indicate whether the image should be inverted (negative)
    '''
    # to run function code from this script
    # file_formats=['.tif','.nd2','.jpg']; segchannel = np.nan; default_columns = ['dataset', 'train_or_test', 'comments']
    # basedirectory = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/'
    # outputdirectory = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/'
    # subdirs = ['Cheeck-Cells_AnnotatedMW_resized/']
    
    # make output directory
    os.makedirs(outputdirectory, exist_ok=True)
    
    # find all .nd2 and .tif files
    allfiles = []
    allfiles_subdirs = []
    for current_subdir in subdirs:
        for current_format in file_formats:
            
            # Collect the file paths
            print(f'Collecting {current_format} files from {basedirectory}')
            current_files = glob.glob(os.path.join(basedirectory, current_subdir, f'*{current_format}'))
            
            # Store their info
            # filename without path
            allfiles.extend([os.path.basename(f) for f in current_files])
            # corresponding subdir
            allfiles_subdirs.extend([current_subdir]*len(current_files))
        
    # Now convert this to a table, with one row per file
    nr_of_rows = len(allfiles) 
    df_metadata = pd.DataFrame({'basedir': nr_of_rows * [basedirectory],
                       'subdir': allfiles_subdirs,
                       'filename': allfiles,
                       'segmentation_channel': nr_of_rows * [segchannel],
                        'invert_image': nr_of_rows * [invert_image]})
    # Now add other columns in default list
    for col in default_columns:
        df_metadata[col] = nr_of_rows * ['']
        
    # now save the metadata file
    outputfilename = os.path.join(outputdirectory, 'metadata_imagefiles_autogen.xlsx')
    df_metadata.to_excel(outputfilename, index=False)
    
    return None
    
    
def get_fileinfo_metadata(df_metadata, file_idx): 
    '''
    extract essential file info from metadata for file with file_idx
    '''
        
    basedir = df_metadata.loc[file_idx, 'basedir']
    subdir = df_metadata.loc[file_idx, 'subdir']
    filename = df_metadata.loc[file_idx, 'filename']
    segchannel = df_metadata.loc[file_idx, 'segmentation_channel']
    do_invert = True if df_metadata.loc[file_idx, 'invert_image']=='yes' else False
    
    return basedir, subdir, filename, segchannel, do_invert
        
    
def invertimage(img):
    '''
    Invert image that is 2d or 3d (per layer)
    '''
    
    if len(img.shape) == 2:
        img_inverted = img.max() - img
    elif len(img.shape) == 3:
        img_inverted = img.max(axis=(1,2))[:,None,None] - img
    else:
        raise ValueError('Image should be 2d or 3d.')
    
    return img_inverted
    
    
def loadsegfile_metadata(df_metadata, file_idx): # , metadatapath=None
    '''
    Loads one image with the to be segmented data, based
    on the metadata file. Specify which image by specifying an 
    index. 
    
    Input is either a pandas metadata dataframe, or 
    lcoation of excel file with that information.
    
    Will either return 2d image if image is 2d or channel for 
    segmentation is given, otherwise, will return 3d image
    (in case of RGB, for example).
    '''
    # 
    
    # Load the data if necessary
    # if df_metadata is None and metadatapath is None:
    #     raise ValueError('Either df_metadata or metadatapath should be given.')
    # if df_metadata is None:
    #     df_metadata = pd.read_excel(metadatapath)
    
    # Get information for file to load
    basedir, subdir, filename, segchannel, do_invert = get_fileinfo_metadata(df_metadata, file_idx)
    
    # Load the file
    fullpath = os.path.join(basedir, subdir, filename)    
    if filename.endswith('.nd2'):
        # nd2 files need separate lib to load
        img = nd2.ND2File(fullpath).asarray()
    else:
        img = skio.imread(fullpath)
        
    # Invert image if desired
    if do_invert:
        img = invertimage(img)
        
    # Return the correct img (channel)
    if segchannel == 'all' or pd.isna(segchannel):
        return img
    else:
        return img[segchannel]
    

    

    