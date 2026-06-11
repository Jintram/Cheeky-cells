
# %%

import numpy as np
import matplotlib.pyplot as plt


# %% 

my_img = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/250920_OY_Batch16_49_tile_img.npy')
plt.imshow(my_img)
# %%

my_img2 = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/250502_OY_09_tile_img.npy')
my_seg2 = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/250502_OY_09_tile_seg.npy')

fig, axs = plt.subplots(1,2)
axs[0].imshow(my_img2)
axs[1].imshow(my_seg2)


# %% Two versions of Yuzeng's files:

img_npy = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/segfiles/20251104batch24_OY_09_seg.npy')
img_npz = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/segfiles/20251104batch24_OY_09_seg.npz')['img_pred_lbls']

# plt.imshow(img_npy)

fig, axs = plt.subplots(1,2)
axs[0].set_title("npy")
axs[0].imshow(img_npy)
axs[1].set_title("npz")
axs[1].imshow(img_npz)



# %%
