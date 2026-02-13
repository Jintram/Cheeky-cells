# %% ################################################################################
# Arabidopsis ML pipeline - Phase 1
#
# Phase 1 goal:
# - create a training set for AI-based segmentation
# - this includes metadata loading and aided annotation generation

from dataclasses import dataclass
import os
import sys

import pandas as pd


# %% ################################################################################
# Phase 1 configuration container (isolated)

@dataclass
class Phase1Config:
    # Root directory of this repository
    script_dir: str = os.path.dirname(os.path.abspath(__file__))

    # Directory with source files used to build metadata
    basedirectory: str = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/'
    subdirs: tuple = ('SELECTION_ML/Cropped/',)

    # Main output directory for analysis products
    outputdirectory: str = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/'

    # Metadata behavior
    regenerate_training_metadata: bool = False
    metadata_customized_filename: str = 'metadata_imagefiles_manual20251022.xlsx'

    # Annotation settings
    tile_size: int = 1000
    bg_percentile: int = 10


# %% ################################################################################
# Phase 1 helpers


def get_output_paths(config: Phase1Config) -> dict:
    # Build output folders used in this phase
    return {
        'segfolder': os.path.join(config.outputdirectory, 'humanseg/'),
        'modelfolder': os.path.join(config.outputdirectory, 'models/'),
        'pltfolder': os.path.join(config.outputdirectory, 'plots/'),
    }


def ensure_output_dirs(paths: dict) -> None:
    # Ensure output directories exist
    os.makedirs(paths['segfolder'], exist_ok=True)
    os.makedirs(paths['modelfolder'], exist_ok=True)
    os.makedirs(paths['pltfolder'], exist_ok=True)


def add_script_dir_to_path(script_dir: str) -> None:
    # Make repository modules importable
    if script_dir not in sys.path:
        sys.path.append(script_dir)


def import_custom_modules_phase1():
    # Import only modules needed for phase 1
    import readwrite.cheeky_readwrite as crw
    import annotating_data.annotation_aided as caa
    import annotating_data.dedicated_segmentation as cds

    return crw, caa, cds


def load_training_metadata(config: Phase1Config, crw) -> pd.DataFrame:
    # Optionally regenerate metadata from raw folders
    if config.regenerate_training_metadata:
        crw.gen_metadatafile(config.basedirectory, list(config.subdirs), config.outputdirectory)

    # Load manually edited metadata file
    metadata_filepath = os.path.join(config.outputdirectory, config.metadata_customized_filename)
    df_metadata = pd.read_excel(metadata_filepath)
    return df_metadata


def create_training_set_from_metadata(df_metadata: pd.DataFrame, config: Phase1Config, output_paths: dict, caa, cds) -> None:
    # Generate segmentation labels and tiles
    caa.annotate_all_pictures_aided(
        df_metadata,
        output_segfolder=output_paths['segfolder'],
        intitial_segfolder=None,
        TILE_SIZE=config.tile_size,
        tile_selection_by='maxvar',
        segfn=cds.basicplantseg1,
        ignore_saved_file=False,
        showplots=False,
        rescalelog=False,
        bg_percentile=config.bg_percentile,
        showrawimg=True,
    )


# %% ################################################################################
# Phase 1 runner


def run_phase1_pipeline(config: Phase1Config) -> None:
    # Prepare path and outputs
    add_script_dir_to_path(config.script_dir)
    output_paths = get_output_paths(config)
    ensure_output_dirs(output_paths)

    # Load custom modules
    crw, caa, cds = import_custom_modules_phase1()

    # Load metadata and generate training labels
    df_metadata = load_training_metadata(config, crw)
    create_training_set_from_metadata(df_metadata, config, output_paths, caa, cds)


# %% ################################################################################
# Execute phase 1 directly

if __name__ == '__main__':
    default_config = Phase1Config()
    run_phase1_pipeline(default_config)
