

# %% ################################################################################
# Some settings

SCRIPT_DIR = '/Users/m.wehrens/Documents/git_repos/_UVA/2025_Cheeky-cells/'

# Directory with image files
basedirectory = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/'
subdirs = ['Cheek-Cells_resized_anonymous/']

# Directory to place output
outputdirectory = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/'

# add scripts in this folder to path
import sys
if SCRIPT_DIR not in sys.path: sys.path.append(SCRIPT_DIR)

# %% ################################################################################

# standard libs
import pandas as pd

import readwrite.cheeky_readwrite as crw
    # reload
    # import importlib; importlib.reload(crw)

import annotating_data.annotation_aided as caa
    # import importlib; importlib.reload(caa)
    
import annotating_data.annotation_postprocessing as cap
    # import importlib; importlib.reload(cap)
    


# %% ################################################################################

# First generate the metadata file
crw.gen_metadatafile(basedirectory, subdirs, outputdirectory)
    # This will auto-generate a metadata file with all image files in the specified folder.
    # It should be manually edited, and stored as an excel file (at metadata_filepath)

# Now update the metadata file location, after editing the auto-generated file and saving under custom name
metadata_customized_filename = 'metadata_imagefiles_MW2025oct-SELECTION.xlsx'
metadata_filepath            = outputdirectory + metadata_customized_filename

# Now load the (manually updated) metadata
df_metadata = pd.read_excel(metadata_filepath)

# %% ################################################################################

# Loop over the files to create the ground truth 
caa.annotate_all_pictures_aided(df_metadata, 
                                output_segfolder=outputdirectory + 'humanseg/',
                                intitial_segfolder=None,
                                TILE_SIZE=1000,
                                tile_selection_by='maxvar',
                                segfn=caa.basicseg2,
                                ignore_saved_file=False,
                                showplots=False,
                                rescalelog=False)


# %% ################################################################################

# Now let's post-process the segmented files to get an improved training mask
segfolder = outputdirectory + 'humanseg/'

# Improve all basic segmentations, ie add boundaries proximity zones
cap.postprocess_basicsegfile_all(df_metadata, segfolder, suffix='_seg_postpr', showplot=True)

# %% ################################################################################
# Now set up a dataset, model, and start training


XXXX
( update `machine_learning/datasetclass/dataset_classes.py` first)
XXXX







# %%
