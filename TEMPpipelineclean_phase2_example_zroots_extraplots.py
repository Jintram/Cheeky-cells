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
        '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/models/modelUNet20260609_1740_stats.npz'
    )['list_confusion_matrices']
list_lr = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/models/modelUNet20260609_1740_stats.npz'
    )['list_lr']

class_names = ['bg','shoot','root','seed','leaf']
custom_colormap2 = {
    class_names[0]: 'black', # bg
    class_names[1]: '#30CC30', # shoot
    class_names[2]: '#666666', # root
    class_names[3]: '#A52A2A', # seed
    class_names[4]: '#006400' # leaf
}

SAVE_DIR_PLOTS = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/plots_training_extra_2hrs/"

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
    0: 'transparent',
    1: '#E9BF94',
    2: 'darkgreen',
    3: 'red'
}

config2 = o2.Phase2Config(
    outputdirectory = '/XXX/',
    metadata_customized_filename = 'metadata_imagefiles_manual.xlsx',
    target_device = 'mps',
    nr_classes = 3,
    img_suffix = '_tile_img_enhanced',
    lbl_suffix = '_tile_seg',
    learning_rate = 1e-3,
    batch_size = 8,
    # for test run (n=100 / e=30 --> 21 mins; .7 mins/epoch)
    # artificial_n=100, 
    # epochs = 30,
    # for actual run
    artificial_n = 1000, # 1000, # ±7min/epoch
    epochs = 120, # 120, # 12 training hours (720 mins) --> ±100 epochs
    lr_schedule_step_len = 40, # should be epochs/3
    cmap_custom = custom_colormap, 
    
    # importantly, load the current model
    model_checkpoint_to_load = 'XXX'
)

dataset_train, dataset_test, model_unet = o2.phase2_setup(config2)

# now load the model parameters
o2.evaluate_on_full_testset_and_plot(config2, model_unet, dataset_test)