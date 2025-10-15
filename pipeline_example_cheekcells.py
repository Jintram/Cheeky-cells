

# %% ################################################################################


import pandas as pd

import readwrite.cheeky_readwrite
    # reload
    # import importlib; importlib.reload(readwrite.cheeky_readwrite)

basedirectory = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/'
outputdirectory = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/'
subdirs = ['Cheek-Cells_resized_anonymous/']

# metadata file location; this file can be partially automatically generated using "gen_metadatafile()" (see below)
metadata_filepath = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/ANALYSIS/20251015/metadata_imagefiles_MW2025oct.xlsx'

# %% ################################################################################


# First generate the metadata file
readwrite.cheeky_readwrite.gen_metadatafile(basedirectory, subdirs, outputdirectory)
    # This will auto-generate a metadata file with all image files in the specified folder.
    # It should be manually edited, and stored as an excel file (at metadata_filepath)

# Now load the (manually updated) metadata
df_metadata = pd.read_excel(metadata_filepath)


