
# %% ################################################################################

# libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# import threshold functions
from skimage.filters import threshold_otsu, threshold_yen, threshold_triangle
import skimage.color as skicolor

from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, disk, binary_opening, binary_closing

# %%

def basicplantseg1(img):
    '''
    very basic segmentation function, used this for root/shoot project.
    '''
    # imgpath = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/SELECTION_ML/Cropped/250920_OY_Batch16_49.tif'    
    # img = np.array(Image.open(imgpath).convert('L')) 
    # img = np.array(Image.open(imgpath))[:,:800,0]
    # plt.imshow(img); plt.show()
        
    thresholdval = threshold_otsu(img)-1
    img_segmaskauto = img > thresholdval
    # plt.imshow(img_segmaskauto); plt.show()
    
    return img_segmaskauto


def basic_planttopview_seg1(img):
    '''
    very basic segmentation function, used this for root/shoot project.
    '''
    # imgpath = '/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/rgb/07_DPI_C01 tray1 7dpi§001_TopRGB_2026-01-15T14-57-00.png.png'
    # img = np.array(Image.open(imgpath))[:,:,:3]
    # plt.imshow(img); plt.show()
    
    # convert the image to HSV
    img_HSV = skicolor.rgb2hsv(img)    
        # plt.imshow(img_HSV)

    # now manual threshold
    sel_mask = \
    (
        # green selection
        (img_HSV[:,:,0] > 50/255) &
        (img_HSV[:,:,0] < 112/255) &
        # value selection
        (img_HSV[:,:,2] > 0.1)
    )
    
    # clean up mask
    # sel_mask_2 = binary_fill_holes(sel_mask)
    sel_mask_2 = binary_opening(sel_mask, footprint=disk(2))
        # closing = dilation followed by erosion
    sel_mask_2 = remove_small_objects(
        sel_mask_2, 
        min_size=100
        )
    
        # plt.imshow(img)
        # plt.imshow(sel_mask_2, alpha=0.7*sel_mask_2)
                
    return sel_mask_2