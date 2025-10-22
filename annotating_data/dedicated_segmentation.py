
# %% ################################################################################

# libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# import threshold functions
from skimage.filters import threshold_otsu, threshold_yen, threshold_triangle

def basicplantseg1(img):
    '''
    very basic segmentation function
    '''
    # imgpath = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/SELECTION_ML/Cropped/250920_OY_Batch16_49.tif'    
    # img = np.array(Image.open(imgpath).convert('L')) 
    # img = np.array(Image.open(imgpath))[:,:800,0]
    # plt.imshow(img); plt.show()
        
    thresholdval = threshold_otsu(img)-1
    img_segmaskauto = img > thresholdval
    # plt.imshow(img_segmaskauto); plt.show()
    
    return img_segmaskauto