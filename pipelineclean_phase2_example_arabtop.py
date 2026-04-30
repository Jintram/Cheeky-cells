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

config2 = o2.Phase2Config(
    outputdirectory = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING_DATA/',
    metadata_customized_filename = 'metadata_imagefiles_manual.xlsx',
    target_device = 'mps',
    nr_classes = 3,
    img_suffix = '_tile_img_enhanced',
    lbl_suffix = '_tile_seg',
    learning_rate = 1e-3,
    batch_size = 8,
    artificial_n = 1000, # lower this during tests
    epochs = 24
)

# %% ###########################################################################
# Running it

saved_model_path = o2.run_phase2_pipeline(config2)
print(f'Model saved to: {saved_model_path}')
# %%
