

import numpy as np
from matplotlib import pyplot as plt

def set_global_fontsize(thesize=8):
    """Set global font size for all plots."""
    plt.rcParams.update({'font.size': thesize})

def overlayplot(current_img_rgb, 
                current_prd, 
                cmap_custom) -> None:
    
    # Save overlay figure with predicted labels over input
    if current_prd.ndim > 2:
        current_pred_lbl = current_prd[0].argmax(0)
    else:
        current_pred_lbl = current_prd

    # Pad to be able to perform the shifted projection
    current_pred_lbl_transl = np.pad(current_pred_lbl, 
                                     ((0, 0), (20, 0)), 
                                     'constant', constant_values=0)
    
    # For plotting appearance
    img_shape_ratio = current_img_rgb.shape[1] / current_img_rgb.shape[0]
    cm_to_inch = 1 / 2.54
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10 * img_shape_ratio * cm_to_inch, 10 * cm_to_inch))
    
    ax.imshow(current_img_rgb)
    ax.imshow(current_pred_lbl_transl, 
              cmap=cmap_custom, 
              vmin=0, vmax=np.max(current_pred_lbl_transl), 
              alpha=1.0 * (current_pred_lbl_transl > 0))
    
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax
    
    
def plot_overlay_n_pred(image, pred, cmap_custom):
    """Plots orginal image and prediction on top, truth to the side"""
    
    fig, axs = plt.subplots(1,2, figsize=(17.2/2.54,(17.2/2)/2.54))  
    
    _ = axs[0].set_title("Original & prediction")
    _ = axs[0].imshow(image)
    _ = axs[0].contour(pred, levels=(np.max(pred)),
                    cmap=cmap_custom,
                    linewidths=.3)

    _ = axs[1].set_title("Prediction")
    _ = axs[1].imshow(pred, cmap=cmap_custom)
    
    # Now save this to the plot folder
    plt.tight_layout()
    
    return fig, axs
    

# def anotherplot_unfinished():
#         fig, ax = plt.subplots(1, 2, figsize=((5 * img_shape_ratio + 5) * cm_to_inch, 5 * cm_to_inch))
#         ax[0].imshow(current_img_rgb)
#         ax[0].imshow(current_pred_lbl_transl, cmap=cmap_plantclasses, vmin=0, vmax=5, alpha=1.0 * (current_pred_lbl_transl > 0))
#         ax[1].imshow(current_lbl, cmap=cmap_plantclasses, vmin=0, vmax=5)
#         ax[0].set_xticks([])
#         ax[0].set_yticks([])
#         ax[1].set_xticks([])
#         ax[1].set_yticks([])