"""
In phase 1, you annotate the images, such that they can be used 
for training and testing purposes.
"""

# %% ###########################################################################
# Libraries

import sys

import orchestrators.orchestrate_phase1_clean as o1
    # import importlib; importlib.reload(o1)
import annotating_data.dedicated_segmentation as cds
    # import importlib; importlib.reload(cds)


# %% ###########################################################################
# Configuration

# Colormap for display of annotation
custom_colormap = {
    0: 'transparent',
    1: 'white',
    2: 'red',
    3: 'green'
}

# Configuration for this dataset
config1 = o1.Phase1Config(
    inputdirectory = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/rgb/',
    outputdirectory = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING_DATA/',
    tile_size = 5000,
    bg_percentile = 10,
    file_formats = ('.png', ),
        # Note: "file_formats" needs to be a "tuple", either ('entry1',) or 
        # ('entry1', 'entry2') etc. Trailing comma required for length 1.
    segfn = cds.basic_planttopview_seg1,
    rescalegreyforseg=False,
    mylabelcolormap=custom_colormap
)

o1.phase1_setup(config1)

# %% ###########################################################################
# Running it

# First, adjust the metadata_imagefiles_autogen.xlsx as required
# (and rename it to avoid overwriting behavior).

config1.metadatafiles_path = \
    "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING_DATA/metadata_imagefiles_manual.xlsx"

# now create annotated pictures
o1.phase1_annotate(config1)
# %%
