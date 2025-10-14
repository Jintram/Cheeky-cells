


# U-Net image segmentation

The goal  of this repository is to facilitate easy training and application of an U-net to 
recognize objects in an image. 

It has been tested and optimized for detection of cells, but can also be used for other 
purposes.

## Workflow description

### Annotating images

For this analysis to work, ground truth images need to be created to train the U-net. 

This requires manual annotation of images, but I've already observed good results with
using only 11 annoted images (which were subdivided over a training and test set), 
that were 2195x2195 pixels in size.

<img src=visuals/annotation_overview.png>
*Overview of annotation procedure. **Left** the original image, **center** the human-generated annotation, and on the **right** training classes that were generated to train the model (this is optional and training can also be done directly on a human-annotated segmentation).*