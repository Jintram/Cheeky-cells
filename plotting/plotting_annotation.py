

# %% ################################################################################

import numpy as np
import matplotlib.pyplot as plt

import plotting_generic as plt_gen 
    # reload 
    # import importlib; importlib.reload(plt_gen)

cm_to_inch = 1/2.54

basedir = '/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/'
basefilename = 'A4_hmKeima_001_tile'

# load the data, ['_transform.npy', '_img.npy', '_img_enhanced.npy',  '_annothuman.npy', '_annothuman_features.npy']
file_list = \
    {'transform': '_transform.npy', 
    'img': '_img.npy', 
    'img_enhanced': '_img_enhanced.npy',  
    'annotation': '_annothuman.npy', 
    'annotation_feat': '_annothuman_features.npy'}

img_data = {}
for key in file_list.keys():
    fullfilename = basedir + basefilename + file_list[key]
    img_data[key] = np.load(fullfilename, allow_pickle=True)
    
 
# Zoom of annotation data
Z_A = [0, 0] + list(img_data['annotation'].shape)  

if False:
    # upper left quadrant
    Z_A = [0, 0] + list(np.ceil(np.array(img_data['annotation'].shape)/2).astype(int))

# Now make a plot
# Set all font sizes to 8 pts
plt.rcParams.update({'font.size': 6})
# The figure
fig, ax = plt.subplots(1, 3, figsize=(20*cm_to_inch, 7*cm_to_inch))
ax[0].imshow(img_data['img_enhanced'], cmap='gray')
ax[0].set_title('Image (contrast corrected)')
#
ax[1].imshow(img_data['img_enhanced'], cmap='viridis')
ax[1].contour(img_data['annotation'], colors='#ff2200', levels=[0.5], linewidths=0.5) # 0066ff
ax[1].set_title('Human annotation')
#
# cmap, norm = plt_gen.colors_to_cmap(plt_gen.color_palette_bangwong_whitefirst)
# ax[2].imshow(img_data['annotation_feat'][Z_A[0]:Z_A[2], Z_A[1]:Z_A[3]], 
#              cmap=cmap, norm=norm, interpolation="nearest") #  vmin=0, vmax=3,
ax[2].imshow(img_data['annotation_feat'][Z_A[0]:Z_A[2], Z_A[1]:Z_A[3]], 
             cmap='viridis', vmin=0, vmax=np.max(img_data['annotation_feat']), interpolation="nearest") #  vmin=0, vmax=3,
ax[2].set_title('Training classes')
plt.tight_layout()
# plt.show()

# Save to pdf
outputdir_plots = basedir + '/_plots/'
fig.savefig(outputdir_plots + basefilename + '_overview-extra2.pdf', bbox_inches='tight', dpi=600)
fig.savefig(outputdir_plots + basefilename + '_overview-extra2.png', bbox_inches='tight', dpi=600)