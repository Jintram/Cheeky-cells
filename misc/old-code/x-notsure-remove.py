

import numpy as np

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, disk, binary_opening, binary_closing
from skimage.measure import label, regionprops

from skimage.filters import median



def preprocess_getbbox_insideplate2(img_in_raw, margin_left = 100, margin_right = 100, 
                                         margin_top = 250, margin_bottom = 250,
                                         min_expected_area = 500000):
    # img_in_raw = img_toseg    
    # margin_left = 100; margin_right = 100; margin_top = 250; margin_bottom = 250; min_expected_area = 500000
    
    # convert to greyscale
    img_in_gray = rgb2gray(img_in_raw)
    # convert to integer, rescale 0.255
    img_in_gray = (img_in_gray/np.max(img_in_gray) * 255).astype(int)
        # plt.hist(img_in_gray.ravel(), bins=50); plt.show()
        # plt.imshow(img_in_gray); plt.show()
    
    # project the image onto 1d both column-wise and row-wise
    img_proj_row = img_in_gray.mean(axis=1)
    img_proj_col = img_in_gray.mean(axis=0)
    # now plot this as line
    plt.plot(img_proj_row/np.max(img_proj_row)); plt.plot(img_proj_col/np.max(img_proj_col)); plt.show()
    
    # get the 1 and 99 percentile values
    p1  = np.percentile(img_in_gray, 1)
    p50 = np.percentile(img_in_gray, 50)
    p99 = np.percentile(img_in_gray, 99)
    
    # identify the background level using the mode
    # bg_level = np.bincount((img_in_gray.ravel()).astype(int)).argmax()
    threshold = p50
        # plt.imshow(img_in_gray<=threshold)
    
    # Create mask
    # mask_otsu = img_in_gray > threshold_otsu(img_in_gray)
    mask = (img_in_gray > threshold).astype(bool)
        # plt.imshow(mask)
        # %matplotlib qt
    
    # Erosion followed by dilation (opening) to remove small bright spots in the background
    # mask = binary_closing(mask, footprint=disk(5))
    # mask = binary_opening(mask, footprint=disk(5))
        # plt.imshow(mask); plt.show()
    
    # Fill all holes
    # mask_filled = binary_fill_holes(mask)
        # plt.imshow(mask_filled); plt.show()
    
    # remove small parts    
    mask_cleaned = remove_small_objects(mask, min_size=10000)
        # plt.imshow(mask_cleaned); plt.show()
    
    # now get bounding box around the mask
    r0, c0, r1, c1 = bbox_from_mask_light(mask_cleaned)
    # regions = regionprops(label(mask_cleaned))
    # largest_region_idx = np.argmax(np.array([r.area for r in regions]))
    # r1, c1, r2, c2 = regions[largest_region_idx].bbox
    
    # If bbox covering minimum area isn't identified, return full image
    if (r1-r0) * (c1-c0) < min_expected_area:
        print("No large enough region detected, returning full image")
        return img_in_raw, (0, img_in_raw.shape[0], 0, img_in_raw.shape[1])
    
    # now create the cropped image
    rect = (r0+margin_top, r1-margin_bottom, c0+margin_left, c1-margin_right)
    img_cropped = img_in_raw[rect[0]:rect[1], rect[2]:rect[3]]
    # plt.imshow(img_cropped); plt.show()
    
    return img_cropped, rect