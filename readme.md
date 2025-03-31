


# Goal of this repository

Students have collected images of cheeck cells, and we want to show how a neural network that segments the pictures might look.

## Data pre-processing

#### Image sizes

We need to know the scale for all pictures, ie to how many microns does a pixel length compare.
This is because a neural network needs to learn about how to recognize cells differently at each scale.

Currently, we have pictures at different magnifications. 
To be able to ues all pictures, I will rescale all pictures to a 20X magnification.
This will either throw away information or create pictures with lacking resolution despite their size.

The function `resize_pictures_infolder(input_folder, output_folder)` will resize all files in the input
folder such that the pixels are consistent with 20X images. 
This means 40X images will be 2X reduced in pixel size, and 10X images will be 2x increased in pixel size.
This functions requires the substring "10X", "20X" or "40X" to be present anywhere the filename, 
such that it can determine the original image size.

#### Image annotation



#### Tiling

(..)