

# %% 

import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import skimage.io as skio
import matplotlib.pyplot as plt

# %matplotlib qt

# %%

# Custom color map with values black, light brown, green (somewhat distinguishable)

# Custom colors 
cmap_custom_aratop = ListedColormap(
    [
        "#000000", # black
        "#9A6324", # brown
        "#3cb44b", # green
    ],
    name="plant_soil_cmap",
) 
    # colors from https://sashamaps.net/docs/resources/20-colors/

# %% 


path_plant_picture = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/images_CR/rgb/14_DPI_C01 tray1 14 dpi§001_TopRGB_2026-01-22T13-32-49.png.png"
path_plant_mask = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/figures/example-plant-mask.tif"
path_soil_mask = "/Users/m.wehrens/Data_UVA/2026_02_araplants_highthroughput/figures/example-soil-mask.tif"

# now load all to np array
plant_picture = skio.imread(path_plant_picture)
plant_mask = (skio.imread(path_plant_mask)>0).astype(int)
soil_mask = (skio.imread(path_soil_mask)>0).astype(int)

# now combine the masks
mask_combined = np.zeros_like(plant_mask)
mask_combined[soil_mask==1] = 1
mask_combined[plant_mask==1] = 2

# zoom region for first plant
y0,y1 = [1100, 1800]
x0,x1 = [150, 800]


# %%

cm_to_inch = 1/2.54
fig, axs = plt.subplots(1, 2, figsize=(10*cm_to_inch, 5*cm_to_inch))

axs[0].set_title("Input image")
axs[0].imshow(plant_picture[y0:y1, x0:x1])
axs[0].axis('off')
axs[1].set_title("Desired output (used for training)")
axs[1].imshow(mask_combined[y0:y1, x0:x1], cmap=cmap_custom_aratop)
axs[1].axis('off')

# %%

plt.imshow(plant_picture[y0:y1, x0:x1])
plt.contour(mask_combined[y0:y1, x0:x1], 
               levels=[0.5, 1.5], 
               colors=["#9A6324", "#3cb44b"])

# %%
