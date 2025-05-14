



import sys; sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/2024_Fluoppi/')
import source_2.fluoppi2_readwrite as flan42_rw
import source.fluoppi_mainlib_v4 as flan4
import source.fluoppi_config as flaconf
import source.fluoppi_morestats as mstats
from source.convenient_definitions import *

import glob
import matplotlib.pyplot as plt
import numpy as np


# This script facilitates converting an nd2 image to the desired format for segmentation
#
# The script needs to integrate with 
#       git_repos/_UVA/2024_Fluoppi/
#       an image can be loaded with "get_my_image_split" in "source_2/fluoppi2_readwrite.py"

TO DO NEXT:
    LOAD AN IMAGE FROM THE LATEST OLAF DATASET (ADD METADATA TABLE FIRST)
    THEN MAKE SURE THIS CAN BE PROCESSED BY THIS FILE


def load_sample_data_for_test():
    
    METADATA_FILE = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/DATA/metadata_Fluoppi_Olaf-20250425.xlsx'
    
    # OUTPUTDIRANALYSIS = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/ANALYSES/analysis_20250425_Olaf/'
    # import os; os.makedirs(OUTPUTDIRANALYSIS, exist_ok=True)
    
    # Now load the metadata
    import pandas as pd
    metadata_table = pd.read_excel(METADATA_FILE, header=0)
    
    # Now load the first image
    img_fluop, img_cells = flan42_rw.get_my_image_split(metadata_table, 0)
    
    # Show
    plt.imshow(np.log(.1+img_cells[2000:4000, 2000:4000])); plt.show(); plt.close()
    
    # Now pre-process this image to 