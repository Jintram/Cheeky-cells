






# Simple code to plot the annotation that was made
# (This code was added later to visualize existing .npy files.)
# (Some plotting is already performed in the other code.)

# import mypytorch_fullseg.dataset_classes_fullseg as ds

#/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg
#/Users/m.wehrens/Data_UVA/2024_07_fluopi_assay/HUMAN_ANNOTATION/20250328_FLUOPPI_humanseg/A4_hmKeima_001_tile_img_enhanced.npy

import numpy as np
import matplotlib.pyplot as plt

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
    
    
# Now make a plot
fig, ax = plt.subplots(1, 3, figsize=(60*cm_to_inch, 20*cm_to_inch))
ax[0].imshow(img_data['img_enhanced'], cmap='gray')
ax[0].set_title('Enhanced image')
ax[1].imshow(img_data['img_enhanced'], cmap='gray')
ax[1].contour(img_data['annotation'], colors='r', levels=[0.5], linewidths=0.5)
ax[1].set_title('Human annotation')
ax[2].imshow(img_data['annotation_feat'], cmap='viridis', vmin=0, vmax=3)
ax[2].set_title('Annotation\n(features)')
plt.tight_layout()
# plt.show()

# Save to pdf
outputdir_plots = basedir + '/_plots/'
fig.savefig(outputdir_plots + basefilename + '_overview-extra.pdf', bbox_inches='tight')