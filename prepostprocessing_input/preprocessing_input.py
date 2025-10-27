#%% ################################################################################
# File to pre-process input

import numpy as np

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

# now do some pre-processing


def preprocess_getbbox_insideplate(img_in_raw, margin_left = 100, margin_right = 100, 
                                         margin_top = 250, margin_bottom = 250):
    # img_in_raw = img_test    
    # margin_left = 100; margin_right = 100; margin_top = 250; margin_bottom = 250
    
    # convert to greyscale
    img_in_gray = rgb2gray(img_in_raw)
        # plt.hist(img_in_gray.ravel()); plt.show()
        # plt.imshow(img_in_gray); plt.show()
    
    # Create mask
    mask_otsu = img_in_gray > threshold_otsu(img_in_gray)
    
    # Fill all holes
    mask_filled = binary_fill_holes(mask_otsu)
        # plt.imshow(mask_filled); plt.show()
    
    # remove small parts    
    mask_cleaned = remove_small_objects(mask_filled, min_size=10000)
        # plt.imshow(mask_cleaned); plt.show()
    
    # now get bounding box of largest object    
    regions = regionprops(label(mask_cleaned))
    largest_region_idx = np.argmax(np.array([r.area for r in regions]))
    r1, c1, r2, c2 = regions[largest_region_idx].bbox
    
    # now create the cropped image
    rect = ((r1+margin_bottom), (r2-margin_top), (c1+margin_left), (c2-margin_right))
    img_cropped = img_in_raw[rect[0]:rect[1], rect[2]:rect[3]]
    # plt.imshow(img_cropped); plt.show()
    
    return img_cropped, rect

    # # now since we know there's no agar at the edges 
    # # and there is agar at the center, we can take those 
    # # values as references..    
    # img_edgevalues_flat = np.concatenate([
    #     img_in_gray[0:10, :].ravel(),
    #     img_in_gray[-10:, :].ravel(),
    #     img_in_gray[:, 0:10].ravel(),
    #     img_in_gray[:, -10:].ravel()])
    
    # (coord1,coord2) = (img_in_gray.shape[0]//3, img_in_gray.shape[0]//3*2)
    # (coord3,coord4) = (img_in_gray.shape[1]//3, img_in_gray.shape[1]//3*2)
    # img_center_values = img_in_gray[coord1:coord2, coord3:coord4].ravel()

    # # for the center, we can take the median
    # center_median = np.median(img_center_values)
    
    # # take the 500th lowest value
    # edge_lowval = np.sort(img_edgevalues_flat)[np.min([500, len(img_edgevalues_flat)-1])]
    
    

    


