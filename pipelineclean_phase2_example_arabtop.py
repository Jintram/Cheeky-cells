"""
In phase 2, the annotated tiles from phase 1 are used to train
a U-Net segmentation model.
"""

# %% ###########################################################################
# Libraries

import cheeky_cells.orchestrators.orchestrate_phase2_WIP as o2
    # import importlib; importlib.reload(o2)


# %% ###########################################################################
# Configuration

# Colormap for display of annotation
custom_colormap = {
    0: 'transparent',
    1: '#E9BF94',
    2: 'darkgreen',
    3: 'red'
}

config2 = o2.Phase2Config(
    outputdirectory = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/',
    metadata_customized_filename = 'metadata_imagefiles_manual.xlsx',
    target_device = 'mps',
    nr_classes = 3,
    img_suffix = '_tile_img_enhanced',
    lbl_suffix = '_tile_seg',
    learning_rate = 1e-3,
    batch_size = 8,
    # for test run
    artificial_n=100, 
    epochs = 30,
    # for actual run
    #artificial_n = 1000, # lower this during tests
    #epochs = 24,
    cmap_custom = custom_colormap
)

dataset_train, dataset_test, model_unet = o2.phase2_setup(config2)

# %% ###########################################################################
# Running it

saved_model_path = o2.phase2_train(config2, dataset_train, dataset_test, model_unet)
print(f'Model saved to: {saved_model_path}')

# %%
