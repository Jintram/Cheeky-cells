"""
Plot training stats using logged training statistics.
"""

"""
# Load a dataset (debuggin only)
npz_content = np.load('/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/TRAINING/TRAINING_2layers/models/modelUNet20260524_2309_stats.npz')
list_confusion_matrices = npz_content['list_confusion_matrices']

class_names = ['background','soil','plant']
"""

################################################################################
# %% 

import os

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

import pandas as pd


################################################################################

def plot_final_confusion_matrix(list_confusion_matrices, 
                                class_names = None,
                                save_path = None):
    """Plot the final confusion matrix from training stats as heatmap.

    A confusion matrix M[C_t, C_p] contains counts for the amount of times
    a true class t was connected to a predicted class p.
    """

    # select last one
    current_matrix = (list_confusion_matrices[-1] if list_confusion_matrices.ndim == 3 else list_confusion_matrices).astype(np.float32)
    
    # normalize
    current_matrix_norm = current_matrix / np.sum(current_matrix, axis=1, keepdims=True)

    # Now plot heatmap
    axs = sns.heatmap(current_matrix_norm*100, cmap='viridis', vmin=0, vmax=100)
    
    # Cosmetics
    axs.set_xlabel("Predicted label")
    axs.set_ylabel("True label")
    if class_names is not None:
        axs.set_xticklabels(class_names)
        axs.set_yticklabels(class_names)
    
    # Color bar cosmetics
    cbar = axs.collections[0].colorbar
    cbar.set_label("Percentage")

    # Plot the values on top
    for i in range(current_matrix_norm.shape[0]):
        for j in range(current_matrix_norm.shape[1]):
            value = current_matrix_norm[i, j]
            text_color = "black" if value > 0.5 else "white"
            axs.text(j + 0.5, i + 0.5, f"{(value*100):.2f}%", ha="center", va="center", color=text_color)
    
    # get fig
    fig = axs.get_figure()
        
    # save if desired
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=600)

    return fig, axs


def plot_IoU(list_confusion_matrices, 
             class_names = None, custom_cmap = None, 
             save_path = None):
    """
    Plot the intersection over union (overlapping part divided by union part) 
    for each class over training epochs.  
    Also include the mean. Accepts custom cmap.
    """
    
    # for readability, go over epochs and classes manually
    IoU_per_class = {cl_idx: [] for cl_idx in range(list_confusion_matrices.shape[1])}
    for epoch in range(len(list_confusion_matrices)):
         for cl_idx in range(list_confusion_matrices.shape[1]):
         
            # get current confusion matrix
            current_matrix = list_confusion_matrices[epoch].astype(np.float32)

            # compute IoU for this class
            TP = current_matrix[cl_idx, cl_idx]
            FP = np.sum(current_matrix[:, cl_idx]) - TP
            FN = np.sum(current_matrix[cl_idx, :]) - TP
            IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
            
            # compute precision
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            MAKE THIS FUNCTION RETURN DATAFRAME, DO PLOTTING OF FAV PARAM LATER
            
            # store IoU in list for plotting
            IoU_per_class[cl_idx].append(IoU)
    
    # convert to pandas df
    df_IoU = pd.DataFrame(IoU_per_class)
    df_IoU['epoch'] = np.arange(len(list_confusion_matrices))
    
    df_IoU_melt = df_IoU.melt(id_vars='epoch', var_name='class', value_name='IoU')
    
    # Set custom names if available
    if class_names is not None:
        df_IoU_melt['class'] = df_IoU_melt['class'].apply(lambda x: class_names[int(x)])
    
    # Now plot using seaborn
    axs = sns.lineplot(
            df_IoU_melt,
            x = "epoch", y = "IoU", 
            hue = "class", 
            palette = custom_cmap)
    axs.set_xlabel("Training progress (epoch)")
    axs.set_ylabel("Intersection over union (Fraction)")
    
    # get fig
    fig = axs.get_figure()
    
    # now save if desired    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=600)
        
    # return
    return fig, axs

def plot_precision(list_confusion_matrices):
    
    
    
     
     
     
     

