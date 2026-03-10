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

import pandas as pd

# custom libs
import readwrite.cheeky_readwrite as crw
    # import importlib; importlib.reload(crw)
import annotating_data.annotation_aided as caa
    # import importlib; importlib.reload(caa)
import annotating_data.dedicated_segmentation as cds

# %% ################################################################################
# Phase 1 configuration container (isolated)

@dataclass
class Phase1Config:

    # Directory with source files used to build metadata
    inputdirectory: str

    # Main output directory for analysis products
    outputdirectory: str

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
    invert_image: str = 'no' 
    default_columns: tuple = ('dataset', 'train_or_test', 'comments')
    
    # metadata file
    metadatafiles_path: str = None
    
    # preliminary segmentation function
    segfn: Callable | None = None
    rescalegreyforseg: bool = True
    
    # custom color map for Napari display
    mylabelcolormap: dict = None


# %% ################################################################################
# Phase 1 runner


def phase1_setup(config1: Phase1Config) -> None:
    
    # Set up output folders
    config1.segfolder = os.path.join(config1.outputdirectory, 'humanseg/')
    config1.pltfolder = os.path.join(config1.outputdirectory, 'plots/')
    os.makedirs(config1.segfolder, exist_ok=True)
    os.makedirs(config1.pltfolder, exist_ok=True)
  
    # Optionally regenerate metadata from raw folders
    if config1.metadatafiles_path is None:
        crw.gen_metadatafile(
            basedirectory=config1.inputdirectory, 
            subdirs=None, 
            outputdirectory=config1.outputdirectory,
            file_formats=config1.file_formats,
            suffix=config1.suffix,
            segchannel = config1.segchannel,
            invert_image = config1.invert_image,
            default_columns=config1.default_columns
        )

def phase1_annotate(config1):
    
    # Load manually edited metadata file
    df_metadata = pd.read_excel(config1.metadatafiles_path)

    # Generate segmentation labels and tiles
    caa.annotate_all_pictures_aided(
        df_metadata,
        output_segfolder=config1.segfolder,
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
        mylabelcolormap=config1.mylabelcolormap
    )
