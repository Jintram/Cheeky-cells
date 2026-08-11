


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

from dataclasses import dataclass, field
from collections.abc import Callable
from matplotlib.colors import ListedColormap

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


from torchvision.transforms import ToTensor

import cheeky_cells.readwrite.cheeky_readwrite as crw
    # import importlib; importlib.reload(crw)
import cheeky_cells.annotating_data.annotation_aided as caa
    # import importlib; importlib.reload(caa)
from cheeky_cells.machine_learning.model import unet_model as cunet


# %% ################################################################################
# Phase 3 configuration container (isolated)

@dataclass
class Phase3Config:
    """Settings for applying a trained model to new images.

    Construct this in your pipeline script and pass it to the phase 3
    functions. Dataset-specific behavior is plugged in via the fn_* fields
    (see documentation/importing_functionality.md).

    Attributes
    ----------
    segmentation_dir : str
        Directory where segmentation output will be put. Holds
        segfiles/<subdir>/, plots/<subdir>/, and log_segmentation.yaml.
        <subdir>/ will mimic the original subdirectories of the data input dir.
    nr_classes : int
        Number of different classes (things) to segment.
    nr_channels_input : int
        Type of input, typically 1 for gray scale, and 3 for color images.
    model_checkpoint_to_load : str
        Path to already trained model (.pth file) to be used for segmentation,
        e.g. <training_dir>/models/modelUNet20251026_1027.pth.
    bg_percentile : int
        Determines how images are normalized before shown to ML network.
        The percentile determines what is considered background, which will
        be subtracted to normalize the image intensity range.
    data_path_input : str
        Path to directory with images to segment. May contain subdirectories
        with images.
    df_metadata : pd.DataFrame | None
        Where metadata of files to segment is stored; populated by
        collect_filelist().
    fn_specific_preprocessing : Callable | None
        Optional preprocessing function that pre-processes all images to be
        segmented. Should look like:
        `img_toseg_prepr, prepr_info = config.fn_specific_preprocessing(img_toseg)`
        Where `img_toseg` and `img_toseg_prepr` are input and output image,
        `prepr_info` is additional information generated that also gets
        stored later in npz.
    fn_plotting : Callable | None
        If set, plots will be made using this function. Should look like:
        `fig, ax = config.fn_plotting(img, pred, cmap, ..)`
        where **config.extraplottingparams will be passed to the function as well.
    target_device : str
        Torch device that the model and image tensors are moved to;
        'mps' (Apple Silicon), 'cuda' (NVIDIA) or 'cpu'. Note that 'cpu' will
        typically work on all machines, but will be very slow. Use 'mps' or
        'cuda' if available.
    cmap_custom : ListedColormap | None
        Custom cmap of type matplotlib.colors.ListedColormap can be provided
        for predicted segmentation masks. If None, a default cmap will be used.
    DPI_plots : int
        Optional; DPI used for plots.
    extraplottingparams : dict
        Optional; extra plotting parameters for a custom plotting function
        can be defined here.
    save_images : bool
        If True, the raw input image and the normalized/enhanced image will be
        saved alongside the segmentation output. This is only useful when the
        segmentation is used to feed back into the annotation loop (phase 1).
    """

    segmentation_dir: str

    # Model settings
    nr_classes: int
    nr_channels_input: int
    model_checkpoint_to_load: str

    # Required image preprocessing settings
    bg_percentile: int

    # Input data metadata settings
    data_path_input: str
    df_metadata: pd.DataFrame | None = None

    # Dataset-specific functions
    fn_specific_preprocessing: Callable | None = None
    fn_plotting: Callable | None = None

    # Model settings with defaults
    target_device: str = 'mps'

    # Optional plotting settings
    cmap_custom: ListedColormap | None = None
    DPI_plots: int = 300
    extraplottingparams: dict = field(default_factory=dict)

    save_images: bool = False


# %% ################################################################################
# Phase 3 helpers

# os.makedirs(paths['segfolder'], exist_ok=True)
# os.makedirs(paths['pltfolder'], exist_ok=True)

def collect_filelist(config: Phase3Config,
                     file_formats=('.tif', '.nd2', '.jpg', '.png'),
                     segchannel='all'):

    config.df_metadata, _ = crw.gen_metadatafile(
        basedirectory=config.data_path_input,
        outputdirectory=config.segmentation_dir,
        file_formats=file_formats,
        segchannel=segchannel,
        save_xlsx=False,
    )

    print("Metadata stored in config.df_metadata.")

    return config


# THIS CONTAINS IMAGE PRE-PROCESSING, AND SHOULD BE DONE SEPARATELY
def get_input_img_file(config: Phase3Config,
                        df_metadata: pd.DataFrame, 
                        file_idx: int):
    
    # Pre-processing info
    prepr_info = None
    
    # Read image from metadata
    img_toseg = crw.loadimgfile_metadata(df_metadata, file_idx,
                                         basedirectory=config.data_path_input,
                                         show_name=True)
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
    
    return img_toseg_prepr_norm, img_toseg_prepr, prepr_info


def initialize_unet_model_for_inference(config: Phase3Config):
    
    ML_model = \
        cunet.UNet(
            n_channels=config.nr_channels_input,
            n_classes=config.nr_classes
        ).to(config.target_device)
    
    # Initialize U-net for inference
    return ML_model


def load_model_checkpoint(model_unet, checkpoint_path: str, target_device: str):

    # Load model weights from checkpoint file; map_location lets you load a
    # checkpoint that was saved on another device (e.g. cuda-trained, applied on mps)
    model_unet.load_state_dict(torch.load(checkpoint_path, map_location=target_device))
    
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
    
    Input arguments;
    Input arguments:
        config: 
            Phase3Config object containing configuration parameters.
        overwrite_files: 
            Boolean indicating whether to overwrite existing segmentation files.
        max_files_to_process: 
            Maximum number of files to process. If None, process all files. 
            Intended for testing purposes.
    
    NOTES
    
    This function could be made much faster if it used batching of images.
    Though already the algorithm takes quite a heavy toll on memory usage.
    
    Fluctuations in memory use can occur due to input images dimensions being 
    different.
    """
        
    # First open the metadata file
    df_metadata_input = config.df_metadata
    if df_metadata_input is None:
        raise ValueError("config.df_metadata is None — call collect_filelist(config) first.")
    
    # Set up model
    print("Initializing U-Net")
    model_unet = initialize_unet_model_for_inference(config)
    model_unet = load_model_checkpoint(model_unet, config.model_checkpoint_to_load, config.target_device)
    model_unet.eval()
    
    # Now make all subdirs that exist, but in the output directory
    print("Creating directory structure")
    for subdir in df_metadata_input['subdir'].unique():
        os.makedirs(os.path.join(config.segmentation_dir, "segfiles/", subdir), exist_ok=True)
        os.makedirs(os.path.join(config.segmentation_dir, "plots/", subdir), exist_ok=True)
    
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
        # file_idx = 8
        # file_idx = 16
        
        print(f'Processing file {file_idx+1}/{nr_files} ..')
        
        # Record start time
        start_time = time.time()
        
        # determine where to store the segfile later
        current_basefilename = os.path.splitext(df_metadata_input.loc[file_idx, "filename"])[0]
        filepath_segfile = \
            os.path.join(config.segmentation_dir, "segfiles/", 
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
        img_toseg_prepr_norm, img_toseg_prepr, prepr_info = get_input_img_file(
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
                **config.extraplottingparams
            )
            
            # save the plot
            fig.savefig(fname = os.path.join(
                                    config.segmentation_dir, "plots/", 
                                    df_metadata_input.loc[file_idx, 'subdir'], 
                                    current_basefilename + "_plot.pdf"), 
                        dpi=config.DPI_plots, bbox_inches='tight')
            plt.close(fig)
        
        # record end time
        end_time = time.time(); time_taken.append(end_time - start_time)
        print("Time taken for this file: {:.2f} seconds".format(end_time - start_time),
              "\nAverage time: {:.2f} seconds".format(np.mean(time_taken)))
        pred_time_remaining = (nr_files - file_idx - 1) * np.mean(time_taken) 
        print(f"Predicted remaining time: {pred_time_remaining/ 60:.2f} minutes")

        # Save raw predicted mask
        print('Saving prediction..')
        np.savez_compressed(filepath_segfile, 
                            img_pred_lbls=img_pred_lbls,
                            prepr_info=prepr_info)
        
        # Optionally save images (for feeding back into annotation loop)
        if config.save_images:
            np.save(filepath_segfile.replace("_seg.npz", "_img.npy"), img_toseg_prepr)
            np.save(filepath_segfile.replace("_seg.npz", "_img_enhanced.npy"), img_toseg_prepr_norm)
        
        print('Saving done..')
        
        # Reset Torch cache to prevent memory build up
        # on my machine, doing this every loop increases speed a load
        if config.target_device == "cuda":
            torch.cuda.empty_cache()
        elif config.target_device == "mps":
            torch.mps.empty_cache()   
                
        files_actually_processed += 1
    
    print("Done")
    
    # Persist resolved config (strip the in-memory DataFrame).
    config_filepath = os.path.join(config.segmentation_dir, "log_segmentation.yaml")
    crw.dump_config_yaml(config, config_filepath, skip_keys=('df_metadata',))
    
    return None

# %%
