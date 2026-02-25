


# %% ################################################################################
# Arabidopsis ML pipeline - Phase 3
#
# This file contains helper and configuration functions that allow you to 
# apply the model to any set of images.
#
# Phase 3 goal:
# - apply trained segmentation model to actual data
# - save overlay plots and predicted masks

# %% ################################################################################
# Libraries

from dataclasses import dataclass
from collections.abc import Callable
from matplotlib.colors import ListedColormap

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


from torchvision.transforms import ToTensor

import readwrite.cheeky_readwrite as crw
    # import importlib; importlib.reload(crw)
import annotating_data.annotation_aided as caa
    # import importlib; importlib.reload(caa)
from machine_learning.model import unet_model as cunet


# %% ################################################################################
# Phase 3 configuration container (isolated)

@dataclass
class Phase3Config:

    # Main output directory
    outputdirectory: str # = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/'

    # Model settings
    nr_classes: int # = 5
    nr_channels_input: int # 3
    
    # Required checkpoint path for inference
    model_checkpoint_to_load: str # = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/models/modelUNet20251026_1027.pth'

    # Required image preprocessing settings
    bg_percentile : int # for intensity normalization, background to subtract

    # Input data metadata settings
    data_path_input: str # '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/'
    metadata_input_filepath: str = None # '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/metadata_imagefiles_manual_inputdata20251027.xlsx'
    
    # Preprocessing function (optional), modifies image
    fn_specific_preprocessing: Callable | None = None
    
    # Function to generate plots
    fn_plotting: Callable | None = None    
    
    # Model settings with defaults
    target_device: str = 'mps'
    
    # Optional plotting settings
    cmap_custom: ListedColormap | None = None


# %% ################################################################################
# Phase 3 helpers

# os.makedirs(paths['segfolder'], exist_ok=True)
# os.makedirs(paths['pltfolder'], exist_ok=True)

def collect_filelist(config: Phase3Config, file_formats=['.tif','.nd2','.jpg','.png'], 
                     segchannel = 'all', invert_image='no'):
    # config = config3_ara_root; file_formats=['.tif','.nd2','.jpg','.png']; segchannel = 'all'; invert_image='no'
    
    config.metadata_input_filepath = \
        crw.gen_metadatafile_segfiles(
                    basedirectory = config.data_path_input, 
                    outputdirectory = config.outputdirectory, 
                    file_formats = file_formats, 
                    segchannel = segchannel, 
                    invert_image = invert_image
        )
    
    return config
    

def load_input_metadata(config: Phase3Config) -> pd.DataFrame:
    
    metadata_filestoseg = pd.read_excel(config.metadata_input_filepath)
    
    return metadata_filestoseg


# THIS CONTAINS IMAGE PRE-PROCESSING, AND SHOULD BE DONE SEPARATELY
def get_input_img_file(config: Phase3Config,
                        df_metadata: pd.DataFrame, 
                        file_idx: int):
    
    # Pre-processing info
    prepr_info = None
    
    # Read image from metadata
    img_toseg = crw.loadimgfile_metadata(df_metadata, file_idx, show_name=True)
        # plt.imshow(img_toseg)

    # Dataset-specific image preprocessor
    if config.fn_specific_preprocessing is not None:
        img_toseg_prepr, prepr_info = config.fn_specific_preprocessing(img_toseg)
    else:
        img_toseg_prepr = img_toseg
        # plt.imshow(img_toseg_prepr)

    # Normalize intensity
    # bg_percentile = 10 is value used for arabidopsis roots
    img_toseg_prepr_norm = crw.image_autorescale(
        img_toseg_prepr, 
        rescalelog=False, 
        bg_percentile=config.bg_percentile)
        # plt.imshow(img_toseg_prepr_norm)
    
    return img_toseg_prepr_norm, prepr_info


def initialize_unet_model_for_inference(config: Phase3Config):
    
    ML_model = \
        cunet.UNet(
            n_channels=config.nr_channels_input,
            n_classes=config.nr_classes
        ).to(config.target_device)
    
    # Initialize U-net for inference
    return ML_model


def load_model_checkpoint(model_unet, checkpoint_path: str):
    
    # Load model weights from checkpoint file
    model_unet.load_state_dict(torch.load(checkpoint_path))
    
    return model_unet


def get_ml_prediction(img_input, the_model, target_device: str, showplot: bool, cmap_plantclasses):
    # Run inference and return predicted class labels
    with torch.no_grad():
        img_torch = ToTensor()(img_input).to(target_device)
        X = img_torch[None, :, :, :]
        logits = the_model(X)
        prd_full = logits.cpu().detach().numpy()
        prd_labels = prd_full[0].argmax(0)

    if showplot:
        plt.imshow(prd_labels, cmap=cmap_plantclasses, vmin=0, vmax=4)
        plt.show()

    return prd_labels

# %% ###########################################################################
# Runner

def segment_all_files(config: Phase3Config,
                      overwrite_files = False,
                      max_files_to_process = None):
    """
    Loads model, and applies it to all images.
    
    This function could be made much faster if it used batching of images.
    Though already the algorithm takes quite a heavy toll on memory usage.
    
    Fluctuations in memory use can occur due to input images dimensions being 
    different.
    """
        
    # First open the metadata file
    df_metadata_input = load_input_metadata(config)
    
    # Set up model
    print("Initializing U-Net")
    model_unet = initialize_unet_model_for_inference(config)
    model_unet = load_model_checkpoint(model_unet, config.model_checkpoint_to_load)
    model_unet.eval()
    
    # Now make all subdirs that exist, but in the output directory
    print("Creating directory structure")
    for subdir in df_metadata_input['subdir'].unique():
        os.makedirs(os.path.join(config.outputdirectory, "segfiles/", subdir), exist_ok=True)
        os.makedirs(os.path.join(config.outputdirectory, "plots/", subdir), exist_ok=True)
    
    # Determine # files to process    
    nr_files = len(df_metadata_input)
    if max_files_to_process is None:
        max_files_to_process = nr_files
    
    print(f"Starting to work on {np.min([nr_files,max_files_to_process])} files..")
               
    # Now go through the dataframe, and produce predictions
    time_taken = []; files_actually_processed = 0
    for file_idx in range(nr_files):
        # file_idx = 0
        # file_idx=454
        
        print(f'Processing file {file_idx+1}/{nr_files} ..')
        
        # Record start time
        start_time = time.time()
        
        # determine where to store the segfile later
        current_basefilename = os.path.splitext(df_metadata_input.loc[file_idx, "filename"])[0]
        filepath_segfile = \
            os.path.join(config.outputdirectory, "segfiles/", 
                         df_metadata_input.loc[file_idx, 'subdir'], 
                         current_basefilename + "_seg.npz")

        # Skip if file was already segged (unless preferred otherwise)
        if not overwrite_files:            
            if os.path.exists(filepath_segfile):
                print(f'Segfile already exists for file idx {file_idx}, skipping..')
                continue
        # Skip if max files to process reached
        if files_actually_processed >= max_files_to_process:
            print(f'Max files to process ({max_files_to_process}) reached, stopping..')
            break
        # Track if the "q" key was pressed, and if so, break
        
        
        # Get image to seg
        img_toseg_prepr_norm, prepr_info = get_input_img_file(
            config = config,
            df_metadata = df_metadata_input,
            file_idx=file_idx
        )
            # plt.imshow(img_toseg_prepr_norm)

        # Get the prediction
        img_pred_lbls = get_ml_prediction(
            img_toseg_prepr_norm,
            the_model=model_unet,
            target_device=config.target_device,
            showplot=False,
            cmap_plantclasses=config.cmap_custom,
        )
            # plt.imshow(img_pred_lbls, cmap = config.cmap_custom)
        
        # Then produce a plot
        print("Segmentation done..")
        if config.fn_plotting is not None:
            
            print("Now plotting")
            fig, ax = config.fn_plotting(
                img_toseg_prepr_norm,
                img_pred_lbls,
                cmap_custom=config.cmap_custom,
            )
            
            # save the plot
            fig.savefig(fname = os.path.join(
                                    config.outputdirectory, "plots/", 
                                    df_metadata_input.loc[file_idx, 'subdir'], 
                                    current_basefilename + "_plot.pdf"), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # record end time
        end_time = time.time(); time_taken.append(end_time - start_time)
        print("Time taken for this file: {:.2f} seconds".format(end_time - start_time),
              "\nAverage time: {:.2f} seconds".format(np.mean(time_taken)))

        # Save raw predicted mask
        print('Saving prediction..')
        np.savez_compressed(filepath_segfile, 
                            img_pred_lbls=img_pred_lbls,
                            prepr_info=prepr_info)
        print('Saving done..')
        
        # Reset Torch cache to prevent memory build up
        # on my machine, doing this every loop increases speed a load
        if config.target_device == "cuda":
            torch.cuda.empty_cache()
        elif config.target_device == "mps":
            torch.mps.empty_cache()   
                
        files_actually_processed += 1
    
    print("Done")
    
    return None

# %%
