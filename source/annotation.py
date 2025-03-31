



from PIL import Image

import napari
import numpy as np

filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized/resizedTo20_Cellen tellen, 40x foto 4.jpg'

# load the image in filepath using PIL
img_PIL = Image.open(filepath)
# convert the image to a np.array
img = np.array(img_PIL)


# Now open the image in Napari
viewer = napari.Viewer()
viewer.add_image(img)
# Add an empty shapes layer for the polygon
polygon_layer = viewer.add_shapes(shape_type='polygon', edge_color='red', face_color='transparent')

napari.run()