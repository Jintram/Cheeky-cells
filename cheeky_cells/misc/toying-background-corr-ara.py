
# %%

# %matplotlib qt

import numpy as np
import matplotlib.pyplot as plt

import cheeky_cells.readwrite.cheeky_readwrite as crw
    # import importlib; importlib.reload(crw)
import cheeky_cells.prepostprocessing_input.ara_roots.preprocessing as prep

img = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/model_seg/segfiles/20250802batch10_OY_06_img.npy")
img2 = crw.image_autorescale(img, rescalelog=False, bg_percentile=50)

the_chann = 2

# determine the median value
mymedian  = np.median(img[:,:,the_chann])
mymedian2 = np.percentile(img[:,:,the_chann], 50)
# determine the mode value
mymode = np.bincount(img[:,:,the_chann].ravel()).argmax()

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(img)#[:,:,the_chann])
ax[0,1].hist(img[:,:,the_chann].ravel(), bins = np.arange(257)-.5)
# plot the mode in the histogram
ax[0,1].axvline(mymedian, color='r', label='median')
ax[0,1].axvline(mymode, color='b',label='mode')
ax[1,0].imshow(img2)#[:,:,the_chann])
ax[1,1].hist(img2[:,:,the_chann].ravel(), bins = np.arange(257)-.5)
plt.show()

plt.imshow(img2)#[:,:,the_chann])


# %% test new functionality

# Note: I wrote this functionality because there were segmentation issues
# due to rescaling issues in taking full sample vs auto-cropped samples.
# And I wanted to be able to generate predictions that are equal to the
# original size. 
# However, turns out tuning the bg_percentile parameter during rescaling
# already resolved the issue. So this functionality has become superfluous.

# First erasing outside-ROI
img3, rect = prep.preprocess_erasebounds(img)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(img)#[:,:,the_chann])
ax[0,1].hist(img[:,:,the_chann].ravel(), bins = np.arange(257)-.5)
# plot the mode in the histogram
ax[0,1].axvline(mymedian, color='r', label='median')
ax[0,1].axvline(mymode, color='b',label='mode')
ax[1,0].imshow(img3)#[:,:,the_chann])
ax[1,1].hist(img3[:,:,the_chann].ravel(), bins = np.arange(257)-.5)
plt.show()

# Then rescaling
img4 = crw.image_autorescale(img3, rescalelog=False, bg_percentile=50)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(img)#[:,:,the_chann])
ax[0,1].hist(img[:,:,the_chann].ravel(), bins = np.arange(257)-.5)
# plot the mode in the histogram
ax[0,1].axvline(mymedian, color='r', label='median')
ax[0,1].axvline(mymode, color='b',label='mode')
ax[1,0].imshow(img4)#[:,:,the_chann])
ax[1,1].hist(img4[:,:,the_chann].ravel(), bins = np.arange(257)-.5)
plt.show()
