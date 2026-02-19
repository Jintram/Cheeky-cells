
# %% ###########################################################################

# Segments arabidopsis root/shoot dataset


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
OUTPUT_DIR = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SEGMENTATION/202602/'
CURRENT_MODEL = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/models/modelUNet20251026_1027.pth'
DATA_DIR = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/'

# Add the script dir to "path"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Now initialize a configuration
config3_ara_root = o3.Phase3Config(
    outputdirectory = OUTPUT_DIR,
    nr_classes = 5,
    nr_channels_input = 3, # (input is rgb, so 3 channels)
    model_checkpoint_to_load = CURRENT_MODEL,
    bg_percentile = 10,
    data_path_input = DATA_DIR,
    fn_specific_preprocessing = pp_ara.preprocess_getbbox_insideplate,
    fn_plotting = pp.overlayplot,
    cmap_custom = plt_ara.cmap_custom_plantclasses
)    

# Collect all files that are to be segmented, store data in metadata
config3_ara_root = o3.collect_filelist(config3_ara_root)
    # config = config3_ara_root
    
    
# now segment them
o3.segment_all_files(config3_ara_root, 
                     max_files_to_process=1, # for test-run purposes
                     overwrite_files=True)



# %%
