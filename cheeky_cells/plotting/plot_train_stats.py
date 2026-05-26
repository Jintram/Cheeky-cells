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
from functools import reduce


################################################################################

def plot_final_confusion_matrix(list_confusion_matrices, 
                                class_names = None,
                                save_path = None,
                                customize_fonts = True,
                                normalize_by = "true"):
    """Plot the final confusion matrix from training stats as heatmap.

    A confusion matrix M[C_t, C_p] contains counts for the amount of times
    a true class t was connected to a predicted class p.
    
    normalize_by is either true, or predicted, that should sum to 1.
    """

    if customize_fonts:
        plt.rcParams.update({'font.size': 7, 'font.family': 'Arial'})

    # select last one
    current_matrix = (list_confusion_matrices[-1] if list_confusion_matrices.ndim == 3 else list_confusion_matrices).astype(np.float32)
    
    # normalize
    if normalize_by == "true":
        current_matrix_norm = current_matrix / (current_matrix.sum(axis=1, keepdims=True) + 1e-8)
    elif normalize_by == "predicted":
        current_matrix_norm = current_matrix / (current_matrix.sum(axis=0, keepdims=True) + 1e-8)
    else:
        raise ValueError("normalize_by should be either 'true' or 'predicted'")
        
    # figure
    fig, axs = plt.subplots(1,1, figsize=(5/2.54, 5/2.54))

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
    
    # add title diagonal = recall or precision, depending on normalization
    if normalize_by == "true":
        axs.set_title("Normalized confusion matrix\n(diagonal = recall)",
                        fontdict={'fontsize': 7})
    elif normalize_by == "predicted":
        axs.set_title("Normalized confusion matrix\n(diagonal = precision)",
                      fontdict={'fontsize': 7})

    # get fig
    fig = axs.get_figure()
        
    # add recall as suffix to save path, assuming path = /path/to/plot.pdf        
    if save_path is not None:
        base, ext = os.path.splitext(save_path)
        if normalize_by == "true":
            save_path = f"{base}_recall{ext}"
        elif normalize_by == "predicted":
            save_path = f"{base}_precision{ext}"
        
    # save if desired
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=600)

    return fig, axs


def get_performance_stats(list_confusion_matrices, 
             class_names = None):
    """
    Plot the intersection over union (overlapping part divided by union part) 
    for each class over training epochs.  
    Also include the mean. Accepts custom cmap.
    """
    
    # for readability, go over epochs and classes manually
    IoU_per_class = {cl_idx: [] for cl_idx in range(list_confusion_matrices.shape[1])}
    precision_per_class = {cl_idx: [] for cl_idx in range(list_confusion_matrices.shape[1])}
    recall_per_class = {cl_idx: [] for cl_idx in range(list_confusion_matrices.shape[1])}
    
    # Calculate multiple performance stats
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
            
            # store IoU in list for plotting
            IoU_per_class[cl_idx].append(IoU)
            precision_per_class[cl_idx].append(precision)
            recall_per_class[cl_idx].append(recall)
    
    # convert each to pandas df
    df_IoU       =  pd.DataFrame(IoU_per_class)
    df_IoU['epoch'] = np.arange(len(list_confusion_matrices))
    df_precision =  pd.DataFrame(precision_per_class)
    df_precision['epoch'] = np.arange(len(list_confusion_matrices))
    df_recall    =  pd.DataFrame(recall_per_class)    
    df_recall['epoch'] = np.arange(len(list_confusion_matrices))
    
    # now combine and melt
    df_stats_long = reduce(lambda left, right: pd.merge(left, right, on=('epoch', 'class')), [
        df_IoU.melt(id_vars = 'epoch', var_name = 'class', value_name='IoU'),
        df_precision.melt(id_vars = 'epoch', var_name = 'class', value_name='Precision'),   
        df_recall.melt(id_vars = 'epoch', var_name = 'class', value_name='Recall')
    ])
    
    if class_names is not None:
        df_stats_long['class'] = df_stats_long['class'].apply(lambda x: class_names[int(x)])

    
    return df_stats_long
                           
def plot_performance(df_stats_long, 
            y,
            custom_cmap = None, 
            save_path = None):
    """ Given df_stats_long, plot performance metric """
    
    # Now plot using seaborn
    axs = sns.lineplot(
            df_IoU_melt,
            x = "epoch", y = y, 
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

# %%

def plot_metrics(list_confusion_matrices, 
                 stats=["IoU", "Precision", "Recall"],
                 class_names = None, 
                 cmap_custom = None,
                 save_path = None, 
                 customize_fonts = True):
    """plot IoU, precision, recall plots"""
    # stats=["IoU", "Precision", "Recall"]; cmap_custom=None
    
    if customize_fonts:
        plt.rcParams.update({'font.size': 7, 'font.family': 'Arial'})
        
    # Get stats
    df_stats_long = get_performance_stats(list_confusion_matrices, 
                                            class_names = class_names)
    
    # Create figure w/ subplots
    fig, axs = plt.subplots(1,len(stats), figsize=(16.7/2.54, 5/2.54))
    
    # Loop over params and plot
    for idx in range(len(stats)):
        sns.lineplot(
            df_stats_long,
            x = "epoch", y = stats[idx], 
            hue = "class", 
            ax = axs[idx],
            palette=cmap_custom,
            linewidth = 1.0)
        # axs[idx].set_yscale("log")
        axs[idx].set_xlabel("Training progress (epoch)")
    
    plt.tight_layout()
    
    # save if desired
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=600)
        
    return fig, axs
    
    
    
    
    
    
     
     
     
     


# %%
