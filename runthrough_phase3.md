


# Description of a typical segmentation run and involved components

## Loading libraries and setting up

This describes document describes what happens if a segmentation run is 
executed. This is referred to as "phase 3" ("phase 1" is annotation of 
training data, and "phase 2" is training the segmentatino network).

Import the 'orchestrator', a script that calls the correct parts
of the scripts in this library.

```
import cheeky_cells.orchestrators.orchestrate_phase3_clean as o3
```

In addition, several plotting functions are supplied in library that
can be used to visualize the end result of the segmentation. 
Import the plotting library:

```
import cheeky_cells.plotting.plotting as pp
```

It is also convenient to define a custom color palette for the output.
This can be done as follows:

```
# Define the colors as an array of hex codes; this will correspond to 
# classes that are segmented
custom_colors_plantclasses = [
    '#000000', # class 0, background, black
    '#90EE90', # class 1, shoot, green
    '#FFFFFF', # class 2, root, white
    '#A52A2A', # class 3, seed, brown
    '#006400', # class 4, leaf, dark green
    '#FF0000', # optional bright red color
]

# now convert to ListedColormap 
from matplotlib.colors import ListedColormap
cmap_custom_plantclasses = ListedColormap(custom_colors_plantclasses)
```

### Configuration

Calling 
```
config3_ara_root = o3.Phase3Config(..)
```
will return an python object that stores parameters that tell
the scripts how to perform the run.

A typical configuration will look as follows:

```
config3_ara_root = o3.Phase3Config(
    segmentation_dir = SEGMENTATION_DIR,
    nr_classes = 5,
    nr_channels_input = 3, # (input is rgb, so 3 channels)
    model_checkpoint_to_load = CURRENT_MODEL,
    bg_percentile = 10,
    data_path_input = DATA_DIR,
    fn_specific_preprocessing = None, # pp_ara.preprocess_getbbox_insideplate2,
    fn_plotting = pp.overlayplot,
    cmap_custom = plt_ara.cmap_custom_plantclasses,
    DPI_plots = 1200
)
```