#%% ################################################################################
# File to pre-process input

import numpy as np

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

# now do some pre-processing


def preprocess_remove_noagar(img_in_raw):
    # img_in_raw = img_test
    
    
    
    # convert to greyscale
    img_in_gray = rgb2gray(img_in_raw)
    plt.hist(img_in_gray.ravel()); plt.show()
        # plt.imshow(img_in_gray); plt.show()
    
    # now since we know there's no agar at the edges 
    # and there is agar at the center, we can take those 
    # values as references..
    
    img_edgevalues_flat = np.concatenate([
        img_in_gray[0:10, :].ravel(),
        img_in_gray[-10:, :].ravel(),
        img_in_gray[:, 0:10].ravel(),
        img_in_gray[:, -10:].ravel()])
    
    (coord1,coord2) = (img_in_gray.shape[0]//3, img_in_gray.shape[0]//3*2)
    (coord3,coord4) = (img_in_gray.shape[1]//3, img_in_gray.shape[1]//3*2)
    img_center_values = img_in_gray[coord1:coord2, coord3:coord4].ravel()
    
    # for the center, we can take the median
    center_median = np.median(img_center_values)
    
    # take the 500th lowest value
    edge_lowval = np.sort(img_edgevalues_flat)[np.min([500, len(img_edgevalues_flat)-1])]
    
    # now show those boundaries on to of histogram
    plt.hist(img_in_gray.ravel()); 
    plt.axvline(x=edge_lowval, color='r')
    plt.axvline(x=center_median, color='r')
    plt.show()
    
    # let's try a threshold at 1/3
    mask_inimg = (img_in_gray > (edge_lowval + (center_median - edge_lowval)/3))
    
    mask_0 = img_in_gray < edge_lowval
    mask_1 = img_in_gray > center_median
    
    plt.imshow(img_in_gray)
    plt.show()
    
    plt.imshow(mask_0)#, colors='r', alpha=mask_inimg*1.0)
    plt.show()
    
    plt.imshow(mask_1)#, colors='r', alpha=mask_inimg*1.0)
    plt.show()
    
    mask_otsu = img_in_gray > threshold_otsu(img_in_gray)
    plt.imshow(mask_otsu)
    plt.show()
    
    # now fill any holes in positive areas within mask_otsu
    from scipy.ndimage import binary_fill_holes
    mask_filled = binary_fill_holes(mask_otsu)
    plt.imshow(mask_filled)

    


