




# %%

import glob 
import pandas as pd
import os 
import numpy as np

dir_files = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAINING-SET-2_20260605/images for annotation-output/segfiles-humancorrected'

# get current list of files
file_list = glob.glob(dir_files + '/*enhanced.npy')
file_list_pd = pd.Series(file_list)

# strip the _img_enhanced.npy
file_list_base = [os.path.basename(X) for X in file_list_pd.str.replace("_img_enhanced.npy", "")]

file_list_base

# %% Compare to other set


files_original = glob.glob('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAINING_202603/images for annotation/*tif')
files_original

files_original_base = [os.path.basename(X).replace(".tif", "") for X in files_original]

# %% now compare the list

# Are all annotated files in the original list?
[np.isin(X, files_original_base) for X in file_list_base]

# And the reverse
[np.isin(X, file_list_base) for X in files_original_base]

# All true; conclusion: these are the same file lists.