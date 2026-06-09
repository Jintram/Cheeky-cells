

# %% ################################################################################

# This file contains a few small functions that aid in loading images, using 
# metadata.


# %% ################################################################################


import os
import shutil
import time
import nd2
import numpy as np
import skimage.io as skio
import glob

import pandas as pd
import yaml

from skimage.exposure import rescale_intensity

# %% ################################################################################
# 

def gen_metadatafile(basedirectory, outputdirectory,
                     file_formats=('.tif', '.nd2', '.jpg'),
                     segchannel='all',
                     save_xlsx=True,
                     output_filename='metadata_imagefiles_autogen.xlsx'):
    '''
    Scan `basedirectory` recursively for image files and build a metadata
    DataFrame with columns: subdir, filename, segmentation_channel,
    train_or_test (the 'train_or_test' column is left empty for the user
    to fill in by hand before phase 2).

    `basedirectory` itself is not stored in the table — it is a per-run
    config value that the caller already knows. `subdir` entries are
    paths relative to `basedirectory`.

    If save_xlsx=True, also writes the table to
    outputdirectory/output_filename.

    Returns (df_metadata, output_filepath_or_None).
    '''

    os.makedirs(outputdirectory, exist_ok=True)

    all_filenames = []
    all_subdirs = []
    for current_format in file_formats:
        print(f'Collecting {current_format} files recursively from {basedirectory}')
        current_paths = glob.glob(os.path.join(basedirectory, f'**/*{current_format}'), recursive=True)
        all_filenames.extend(os.path.basename(p) for p in current_paths)
        all_subdirs.extend(os.path.relpath(os.path.dirname(p), basedirectory) for p in current_paths)

    nr_of_rows = len(all_filenames)
    df_metadata = pd.DataFrame({
        'subdir': all_subdirs,
        'filename': all_filenames,
        'segmentation_channel': [segchannel] * nr_of_rows,
        'train_or_test': [''] * nr_of_rows,
    })

    output_filepath = None
    if save_xlsx:
        output_filepath = os.path.join(outputdirectory, output_filename)
        df_metadata.to_excel(output_filepath, index=False)

    return df_metadata, output_filepath

    
def get_fileinfo_metadata(df_metadata, file_idx):
    '''
    Extract essential file info from metadata for file with file_idx.
    Returns: subdir, filename, segchannel.
    (`basedir` is no longer stored per-row; pass it explicitly to loaders.)
    '''

    subdir = df_metadata.loc[file_idx, 'subdir']
    filename = df_metadata.loc[file_idx, 'filename']
    segchannel = df_metadata.loc[file_idx, 'segmentation_channel']

    # some contingencies for if subdir is either number or nan
    if pd.isna(subdir):
        subdir = ''
    elif isinstance(subdir, (int, float)):
        subdir = str(subdir)

    return subdir, filename, segchannel
        
    
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
    
    
def loadimgfile_metadata(df_metadata, file_idx, basedirectory, show_name=False, do_invert=False):
    '''
    Loads one image with the to be segmented data, based
    on the metadata file. Specify which image by specifying an
    index.

    `basedirectory` is supplied by the caller (typically a Phase config) —
    it is not stored per-row in the metadata table.

    Will either return 2d image if image is 2d or channel for
    segmentation is given, otherwise, will return 3d image
    (in case of RGB, for example).

    Set do_invert=True to return the negative of the image. Inversion is
    dataset-wide rather than per-file, so callers decide.
    '''

    # Get information for file to load
    subdir, filename, segchannel = get_fileinfo_metadata(df_metadata, file_idx)

    if show_name:
        print(f"Loading file {filename}..")
    
    # Load the file
    fullpath = os.path.join(basedirectory, subdir, filename)    
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
    _, filename, _ = get_fileinfo_metadata(df_metadata, file_idx)

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
    _, filename, _ = get_fileinfo_metadata(df_metadata, file_idx)
    
    # determine new filename
    filename_base = os.path.splitext(filename)[0]
    newfilename_annot        = filename_base + suffix + '.npy'
    
    # now write the file
    np.save(segfolder + newfilename_annot, x)
    
# %% ################################################################################
# Provenance / logging helpers shared across phases


def _yaml_safe(v):
    # Convert one value into something pyyaml can serialize cleanly.
    # Callables get stringified as "<callable mod.qualname>"; numpy scalars/arrays
    # are coerced to native types; anything unknown falls back to repr().
    
    # keep "normal" classes
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    # convert if non-compatible (e.g. functions)
    if callable(v):
        mod = getattr(v, '__module__', '') or ''
        qn = getattr(v, '__qualname__', getattr(v, '__name__', repr(v)))
        return f'<callable {mod}.{qn}>'
    # recursive for nested stuff
    if isinstance(v, (list, tuple)):
        return [_yaml_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _yaml_safe(x) for k, x in v.items()}
    # also keep "normal" np classes
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    
    return repr(v)


def dump_config_yaml(config, output_path, skip_keys=()):
    '''
    Dump a dataclass-like config object's attributes to a YAML file. Callables
    are recorded as 'module.qualname' strings; numpy types are normalized;
    anything else unrepresentable is repr()'d. Pass attribute names in
    `skip_keys` to exclude them (e.g. an in-memory DataFrame).
    '''
    # convert to compatible output format
    skip = set(skip_keys)
    out = {k: _yaml_safe(v) for k, v in vars(config).items() if k not in skip}
    # and save to disk
    with open(output_path, 'w') as f:
        yaml.dump(out, f, sort_keys=False)
    return output_path


def write_training_images_filelist(df_metadata, output_path, inputdirectory):
    '''
    Write a human-readable provenance snapshot of the training images listed in
    `df_metadata` (columns: subdir, filename). The output is plain text, sorted,
    with a short header recording the absolute `inputdirectory` they came from.
    Reconstruct absolute paths by joining `inputdirectory` with each line.
    '''
        
    with open(output_path, 'w') as f:
        
        f.write(f'# inputdirectory: {os.path.abspath(inputdirectory)}\n')
        f.write(f'# generated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# total: {len(df_metadata)} files\n')
        
        for subdir, filename in zip(df_metadata['subdir'], df_metadata['filename']):
            
            if pd.isna(subdir): 
                subdir = ''
            
            if subdir and subdir != '.':
                f.write(f'{subdir}/{filename}\n')
            else:
                f.write(f'{filename}\n')
                
    return output_path


def copy_originals_for_training(df_metadata, src_inputdirectory, dst_originals_dir):
    '''
    Copy the image files listed in `df_metadata` from `src_inputdirectory`
    into `dst_originals_dir`, preserving the `subdir` layout. Skips files that
    already exist at the destination (idempotent). Prints one line per file
    that is skipped or fails to copy.
    '''

    os.makedirs(dst_originals_dir, exist_ok=True)
    for _, row in df_metadata.iterrows():
        subdir = '' if pd.isna(row['subdir']) else str(row['subdir'])
        filename = str(row['filename'])
        subdir_filename = os.path.join(subdir, filename) if subdir and subdir != '.' else filename
        src = os.path.join(src_inputdirectory, subdir_filename)
        dst = os.path.join(dst_originals_dir, subdir_filename)
        if os.path.exists(dst):
            print(f'  [skip — exists] {subdir_filename}')
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        except OSError as e:
            print(f'  [copy failed] {subdir_filename}: {e}')


# %%
# 