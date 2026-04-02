


# Some notes regarding procedures for the different datasets

### Fluorescent cells

Data was created using very large images of fluorescent cells. 
(E.g. 10.000 x 10.000 pixels, because they were stitched microscopy images.) 
These were tiled into tiles of X times X pixels, and I think I selected 
the tile with the most variance (~most cells) for annotation.

### Arabidopsis shoots (sideview)

Preliminary training data was custom-cropped, such that file dimensions
were managable (anyways, the original images were a reasonable size, e.g. 
3277 x 3181 pixels), and a good view of plants was obtained. (This
perhaps also resulted in artifacts not being recognized properly, 
as they were not included in the training data.)

### Arabidopsis plants (top view)

Preliminary data consists of rather large images.
Full-sized images were annotated.