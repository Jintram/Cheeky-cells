
# %% ###########################################################################

# Segments arabidopsis root/shoot dataset

# In this example, also image and rescaled images are outputted
# Now, the output can be used for training purposes (ie serve as input
# for phase 1).

# %% ###########################################################################
# Libraries

import sys

import orchestrators.orchestrate_phase3_clean as o3
    # import importlib; importlib.reload(o3)
    # import importlib; importlib.reload(crw)

# Dataset-specific imports
# To pre-process a raw image
import prepostprocessing_input.ara_roots.preprocessing as pp_ara
import prepostprocessing_input.ara_roots.ara_plotting as plt_ara
    # import importlib; importlib.reload(pp_ara)

import plotting.plotting as pp
    # import importlib; importlib.reload(pp)

# %% ###########################################################################
# Configuration

# general config
SCRIPT_DIR = '/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-MolCyto/2025_Cheeky-cells/'

# dataset spcecific config
CURRENT_MODEL = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/models/modelUNet20251026_1027.pth'
DATA_DIR = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/Originals/'
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/model_seg/'

# Add the script dir to "path"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Now initialize a configuration
# The flag "save_images=True" generates additional output data.
# This allows the segmented images to be used as training data.
# Turn this of if you don't want this, to avoid excessive hard disk use.
config3_ara_root = o3.Phase3Config(
    outputdirectory = OUTPUT_DIR,
    nr_classes = 5,
    nr_channels_input = 3, # (input is rgb, so 3 channels)
    model_checkpoint_to_load = CURRENT_MODEL,
    bg_percentile = 10,
    data_path_input = DATA_DIR,
    fn_specific_preprocessing = None,
        # Let's not do cropping
    fn_plotting = pp.overlayplot,
    cmap_custom = plt_ara.cmap_custom_plantclasses,
    save_images=True
        # Required to export to 
)    

# Collect all files that are to be segmented, store data in metadata
config3_ara_root = o3.collect_filelist(config3_ara_root)
    # config = config3_ara_root
    
    # hacky option (do not use)
    # file_formats =["_img.npy"]
    
    
# now segment them
o3.segment_all_files(config3_ara_root,
                     # max_files_to_process=200, # for test-run purposes
                     overwrite_files=True
                     )


# %%
# DEBUGGING

if False:
    
    # Given a filename (e.g. 20250620_OY_06), find index in the df_metadata_input
    np.where(df_metadata_input.loc[:,'filename'].str.contains('20250617_OY_17'))
    np.where(df_metadata_input.loc[:,'filename'].str.contains('20250617_OY_15'))
    
