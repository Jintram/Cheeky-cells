

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

from skimage.exposure import rescale_intensity

# %% ################################################################################
# 

def gen_metadatafile(basedirectory, subdirs, outputdirectory, 
                     suffix='',
                     file_formats=['.tif','.nd2','.jpg'], 
                     segchannel = 'all', invert_image='no', 
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
    
    # set subdirs if not given
    if subdirs is None:
        subdirs = ['']
    
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
    outputfilename = os.path.join(outputdirectory, 'metadata_imagefiles_autogen'+suffix+'.xlsx')
    df_metadata.to_excel(outputfilename, index=False)
    
    return None


def gen_metadatafile_segfiles(basedirectory, outputdirectory, 
                     file_formats=['.tif','.nd2','.jpg','.png'], 
                     segchannel = 'all', invert_image='no'):
    '''
    Given a folder, generate an excel file with column 'filename' with all image files
    from a given directory and its subdirectories, and add also default columns
    that the user can fill in further.
    
    Invert_image can be 'yes' or 'no', to indicate whether the image should be inverted (negative)
    '''
    
    # make output directory
    os.makedirs(outputdirectory, exist_ok=True)
    
    # find all image files
    all_paths = []
    for ext in file_formats:
        all_paths.extend(glob.glob(os.path.join(basedirectory,f'**/*{ext}'), recursive=True))
    
    # Now get the subdirs and filenames
    all_subdirs = [
        os.path.relpath(os.path.dirname(current_path), basedirectory) 
            for current_path in all_paths
        ]
    all_filenames = [
        os.path.basename(current_path) 
            for current_path in all_paths
        ]
                        
    # Now convert this to a table, with one row per file
    nr_of_rows = len(all_filenames) 
    df_metadata = pd.DataFrame({'basedir': nr_of_rows * [basedirectory],
                       'subdir': all_subdirs,
                       'filename': all_filenames,
                       'segmentation_channel': nr_of_rows * [segchannel],
                       'invert_image': nr_of_rows * [invert_image]})
                        # os.path.join(basedirectory, all_subdirs[0], all_filenames[0])
        
    # now save the metadata file
    metadata_toseg_filepath = os.path.join(outputdirectory, 'metadata_files_toseg_autogen.xlsx')
    df_metadata.to_excel(metadata_toseg_filepath, index=False)
    
    return metadata_toseg_filepath

    
def get_fileinfo_metadata(df_metadata, file_idx): 
    '''
    extract essential file info from metadata for file with file_idx
    '''
        
    basedir = df_metadata.loc[file_idx, 'basedir']
    subdir = df_metadata.loc[file_idx, 'subdir']
    filename = df_metadata.loc[file_idx, 'filename']
    segchannel = df_metadata.loc[file_idx, 'segmentation_channel']
    do_invert = True if df_metadata.loc[file_idx, 'invert_image']=='yes' else False
    
    # some contingencies for if subdir is either number or nan
    if pd.isna(subdir):
        subdir = ''
    elif isinstance(subdir, (int, float)):
        subdir = str(subdir)
    
    return basedir, subdir, filename, segchannel, do_invert
        
    
def invertimage(img):
    '''
    Invert image that is 2d or 3d (per layer)
    '''
    
    img_inverted = img.max() - img
    
    # if len(img.shape) == 2:
    #     img_inverted = img.max() - img
    # elif len(img.shape) == 3:
    #     img_inverted = img.max(axis=(1,2))[:,None,None] - img
    # else:
    #     raise ValueError('Image should be 2d or 3d.')
    
    return img_inverted
    
    
def loadimgfile_metadata(df_metadata, file_idx, show_name=False): # , metadatapath=None
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
    
    if show_name:
        print(f"Loading file {filename}..")
    
    # Load the file
    fullpath = os.path.join(basedir, subdir, filename)    
    if filename.endswith('.nd2'):
        # nd2 files need separate lib to load
        img = nd2.ND2File(fullpath).asarray()
    elif filename.endswith('.npy'):
        img = np.load(fullpath)
    else:
        img = skio.imread(fullpath)
        # plt.imshow(img); plt.show()
        
    # Invert image if desired
    if do_invert:
        img = invertimage(img)
        
    # reduce channels to 3 if we have more (png might have alpha channel, e.g.)
    if len(img.shape) == 3:
        if img.shape[2] > 3:
            img = img[:,:,:3]     
        
    # Return the correct img (channel)
    if segchannel == 'all' or pd.isna(segchannel):
        return img
    else:
        return img[segchannel]
    

def subtractbaseline(anarray, bg_percentile=.2):
    
    # prevent modifying original array
    anarray = anarray.copy()
    
    # apply subtraction
    thresholdlow = np.percentile(anarray, bg_percentile)
    anarray[anarray < thresholdlow] = thresholdlow
    anarray = anarray - thresholdlow
    
    return anarray

def image_autorescale(input_img, rescalelog=True, bg_percentile=.2):
    """
    bg_percentile between 0..100
    """
    # input_img=np.zeros([20,20])
    # input_img=img_toseg
    
    # plt.hist(input_img.ravel())
    # plt.hist(image_autorescale(input_img).ravel())
    
    # make the image float32 first
    input_img = input_img.copy().astype(np.float32)
    
    # first add a 3rd dimension (corresponding to channels) if there isn't any
    if len(input_img.shape)==2:
        input_img = input_img[:,:,np.newaxis]
    
    # now go over each channel and perform rescaling
    new_img = np.zeros_like(input_img)
    for ch_idx in range(input_img.shape[2]): # ch_idx = 0 
        if rescalelog:  
            new_img[:,:,ch_idx] = np.log(.1+subtractbaseline(input_img[:,:,ch_idx], bg_percentile=bg_percentile))
        else:
            new_img[:,:,ch_idx] = subtractbaseline(input_img[:,:,ch_idx], bg_percentile=bg_percentile)
        
    # rescale to range 0-255
    new_img = rescale_intensity(new_img, 'image', (0, 255))
    new_img = new_img.astype(np.uint8)
        # plt.imshow(new_img); plt.show()
    
    return new_img    
    
def addsuffixtoonefilename(filename, suffix, newextension='.npy', extensionlist=['.npy', '.tif', '.jpg', '.png', '.nd2']):
    '''
    Add a suffix to a filename, replace the extension with another one if desired.
    Default: change to .npy extension, set to None to keep original.    
    
    Will throw warning if original extension not in extensionlist.
    '''
    
    # separate basename and extension
    filename_base, filename_ext = os.path.splitext(filename)
    
    if newextension is not None:
        filename_ext = newextension
    
    # Generate new filename    
    newfilename = filename_base + suffix + filename_ext        
    
    # Raise warning if extension not in recognized list
    if filename_ext not in extensionlist:
        print(f'Warning: extension {filename_ext} not in recognized extension list.')
        
    return newfilename

def addsuffixtofilenames(filename, suffix, newextension='.npy', extensionlist=['.npy', '.tif', '.jpg', '.png', '.nd2']):
    '''
    Will apply addsuffixtoonefilename to a single input or array.
    (That function adds suffixes to a filename and replaces extension if desired.)
    '''
    
    # Convert to array if necessary
    filename = np.array(filename)
    if len(filename.shape) == 0:
        filename = filename.reshape(1)
    
    # Now apply to addsuffixtoonefilename()
    newfilenames = \
        [addsuffixtoonefilename(filename, suffix, newextension=newextension, extensionlist=extensionlist) for
            filename in filename]
         
    return newfilenames


def loadsegfile_metadata(df_metadata, file_idx, segfolder, suffix='_seg', silence=True):
    '''
    Looks for a segmentation mask file in the form of _seg.npy or _seg.tif based
    on df_metadata, file_idx. 
    Requirement is that it has the same resolution as the original files, and is
    a binary mask.
    
    These can both be externally provided segmentation masks or segmentation masks
    that were generated within this script collection.
    '''
    
    # Set None value that will be returned in case no img found
    img_annot = None
    
    # Get image info
    _, _, filename, _, _ = get_fileinfo_metadata(df_metadata, file_idx)
    
    # Get current file extension
    filename_extension = os.path.splitext(filename)[1]
    
    # Check for existing seg file
    if not silence:
        print('Looking for', filename.replace(filename_extension, suffix+'.npy'), ',', 
                             filename.replace(filename_extension, suffix+'.tif'))
    filepath_npy = os.path.join(segfolder, filename.replace(filename_extension, suffix+'.npy'))
    # filepath_npz = os.path.join(segfolder, filename.replace(filename_extension, suffix+'.npz'))
    filepath_tif = os.path.join(segfolder, filename.replace(filename_extension, suffix+'.tif'))
    
    # if os.path.exists(filepath_npy) and os.path.exists(filepath_npz):
    #     raise ValueError("Both .npy and .npz seg files found for this image. Please remove one of them.")
        
    # Load the file
    if os.path.exists(filepath_npy):
        img_annot = np.load(filepath_npy)
    # elif os.path.exists(filepath_npz):
    #     img_annot = np.load(filepath_npz)['image']
    elif os.path.exists(filepath_tif):
        img_annot = skio.imread(filepath_tif)
        
    # Flatten to 2d
    if not img_annot is None:
        if len(img_annot.shape) == 3:            
            img_annot = img_annot.max(axis=2)
    
    if (img_annot is None) and (not silence):
        print('No seg file found..')
    
    # Return it
    return img_annot    


def savesegfile_default(x, df_metadata, file_idx, segfolder, suffix):
    '''
    Saving data "x" to a npy file using standard filename formatting
    for segfiles.
    '''
    
    # determine filename from metadata
    _, _, filename, _, _ = get_fileinfo_metadata(df_metadata, file_idx)
    
    # determine new filename
    filename_base = os.path.splitext(filename)[0]
    newfilename_annot        = filename_base + suffix + '.npy'
    
    # now write the file
    np.save(segfolder + newfilename_annot, x)
    
# %%
# 