"""
Re-generate performance plots in a custom way. 
(These plots are already made during training as well.)
"""

# %% ###########################################################################
# Libraries

import numpy as np
import cheeky_cells.plotting.plot_train_stats as ptt
    # import importlib; importlib.reload(ptt)
import cheeky_cells.orchestrators.orchestrate_phase2_clean as o2
    # import importlib; importlib.reload(o2)
import pandas as pd
import imageio

# %% ###########################################################################
# PERFORMANCE PLOTS
# These plots are already made automatically, but can be further 
# customized here if desired.

    # import importlib; importlib.reload(ptt)

list_confusion_matrices = np.load(
        '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/models/modelUNet20260610_2228_stats.npz'
    )['list_confusion_matrices']
list_lr = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/models/modelUNet20260610_2228_stats.npz'
    )['list_lr']

class_names = ['bg','shoot','root','seed','leaf']
custom_colormap2 = {
    class_names[0]: 'black', # bg
    class_names[1]: '#30CC30', # shoot
    class_names[2]: '#666666', # root
    class_names[3]: '#A52A2A', # seed
    class_names[4]: '#006400' # leaf
}

SAVE_DIR_PLOTS = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/plots_training_extra/"

_, _ = ptt.plot_metrics(list_confusion_matrices, 
                 cmap_custom = custom_colormap2,
                 class_names=class_names,
                 save_path = SAVE_DIR_PLOTS+\
                            'performance_during_training.pdf',
                myheight=7/2.54)

_, _ = ptt.plot_final_confusion_matrix(list_confusion_matrices, 
                                class_names = class_names,
                                save_path = SAVE_DIR_PLOTS+
                            'confusion_matrix.pdf')

_, _ = ptt.plot_final_confusion_matrix(list_confusion_matrices, 
                                class_names = class_names,
                                normalize_by = "predicted",
                                save_path = SAVE_DIR_PLOTS+
                            'confusion_matrix.pdf')

_, _ = ptt.plot_learning_rate(list_lr,
                            save_path = SAVE_DIR_PLOTS+
                            'learning_rate.pdf')


################################################################################
# %% Also generate side-by-side images of the complete test image


# Colormap for display of annotation
custom_colormap = { 
    0: 'transparent', # bg
    1: '#30CC30', # shoot
    2: '#666666', # root
    3: '#A52A2A', # seed
    4: '#006400' # leaf
}

config2 = o2.Phase2Config(
    training_dir = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/',
    metadata_customized_filename = 'metadata_imagefiles_manual.xlsx',
    target_device = 'mps',
    nr_classes = 5,
    img_suffix = '_img_enhanced',
    lbl_suffix = '_seg',
    learning_rate = 1e-3,
    batch_size = 8,
    # for test run
    artificial_n=5000, # 1000
    epochs = 24, # 24
    # for actual run
    #artificial_n = 1000, # lower this during tests
    #epochs = 24,
    cmap_custom = custom_colormap, 
    
    # importantly, load the current model
    model_checkpoint_to_load = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/models/modelUNet20260610_2228__trained13h01m.pth'
)

dataset_train, dataset_test, model_unet = o2.phase2_setup(config2)

# now load the model parameters
o2.evaluate_on_full_testset_and_plot(config2, model_unet, dataset_test)
# %%
