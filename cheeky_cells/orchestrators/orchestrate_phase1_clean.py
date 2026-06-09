# %% ################################################################################
# Arabidopsis ML pipeline - Phase 1
#
# Phase 1 goal:
# - create a training set for AI-based segmentation
# - this includes metadata loading and aided annotation generation

from dataclasses import dataclass
from collections.abc import Callable
import os
import sys
import time

import pandas as pd

# custom libs
import cheeky_cells.readwrite.cheeky_readwrite as crw
    # import importlib; importlib.reload(crw)
import cheeky_cells.annotating_data.annotation_aided as caa
    # import importlib; importlib.reload(caa)
import cheeky_cells.annotating_data.dedicated_segmentation as cds

# %% ################################################################################
# Phase 1 configuration container (isolated)

@dataclass
class Phase1Config:

    # Directory with source files used to build metadata
    inputdirectory: str

    # Training-session directory: holds humanseg/, plots/, metadata, logs,
    # and (optionally) originals/. Phase 2 reads from the same directory.
    training_dir: str

    # Annotation settings
    tile_size: int = 1000
    bg_percentile: int = 10
    
    # Output folders
    segfolder: str = None
    pltfolder: str = None
    
    # metadata configuration optoins
    suffix: str = ''
    file_formats: tuple = ('.tif', '.nd2', '.jpg')
    segchannel: str = 'all'

    # metadata file
    metadatafiles_path: str = None
    
    # preliminary segmentation function
    segfn: Callable | None = None
    rescalegreyforseg: bool = True
    
    # custom color map for Napari display
    mylabelcolormap: dict = None
    # custom Napari function
    my_napari_function: Callable | None = None
    
    # Infix for output filenames (default '_tile' from tiling;
    # set to '' when correcting phase 3 output)
    file_infix: str = '_tile'

    # Copy originals listed in metadata into training_dir/originals/ and
    # repoint inputdirectory there. Off by default: keeps training_dir small.
    copy_originals: bool = False


# %% ################################################################################
# Phase 1 runner


def phase1_setup(config1: Phase1Config) -> None:
    
    # Set up output folders
    if config1.segfolder is None:
        config1.segfolder = os.path.join(config1.training_dir, 'humanseg/')
    if config1.pltfolder is None:    
        config1.pltfolder = os.path.join(config1.training_dir, 'plots/')
    os.makedirs(config1.segfolder, exist_ok=True)
    os.makedirs(config1.pltfolder, exist_ok=True)
  
    # Optionally regenerate metadata from raw folders
    if config1.metadatafiles_path is None:
        crw.gen_metadatafile(
            basedirectory=config1.inputdirectory,
            outputdirectory=config1.training_dir,
            file_formats=config1.file_formats,
            segchannel=config1.segchannel,
            save_xlsx=True,
            output_filename=f'metadata_imagefiles_autogen{config1.suffix}.xlsx',
        )

def phase1_annotate(config1):
    
    # Load manually edited metadata file
    df_metadata = pd.read_excel(config1.metadatafiles_path)

    # Provenance snapshot — which originals does this training session use?
    crw.write_training_images_filelist(
        df_metadata,
        os.path.join(config1.training_dir, 'training_images_filelist.txt'),
        config1.inputdirectory,
    )

    # Optionally stage originals into training_dir/originals/ for self-contained
    # training sessions; repoint inputdirectory so the rest of phase 1 reads
    # from the local copy.
    if config1.copy_originals:
        originals_dir = os.path.join(config1.training_dir, 'originals/')
        print(f'Copying original files to: {originals_dir}')
        crw.copy_originals_for_training(df_metadata, config1.inputdirectory, originals_dir)
        config1.inputdirectory = originals_dir

    # Generate segmentation labels and tiles
    caa.annotate_all_pictures_aided(
        df_metadata,
        output_segfolder=config1.segfolder,
        basedirectory=config1.inputdirectory,
        intitial_segfolder=None,
        TILE_SIZE=config1.tile_size,
        tile_selection_by='maxvar',
        segfn=config1.segfn,
        ignore_saved_file=False,
        showplots=False,
        rescalelog=False,
        bg_percentile=config1.bg_percentile,
        showrawimg=True,
        rescalegrey=config1.rescalegreyforseg,
        mylabelcolormap=config1.mylabelcolormap,
        mynaparifunction=config1.my_napari_function,
        file_infix=config1.file_infix
    )

    # Persist resolved config (after any inputdirectory repointing) for traceability.
    log_path = os.path.join(config1.training_dir, f'phase1_log_{time.strftime("%Y%m%d_%H%M%S")}.yaml')
    crw.dump_config_yaml(config1, log_path)
    print(f'Phase 1 log written: {log_path}')
