


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

A typical configuration will look as the following example:

```
config3_ara_root = o3.Phase3Config(
    segmentation_dir = \
        '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/SEG_2026_highresmodel-crop_TESTSET/',
    nr_classes = 5,
    nr_channels_input = 3, # (input is rgb, so 3 channels)
    model_checkpoint_to_load = \
        '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618_cleaned/models/modelUNet20260619_2100__trained0d19h46m.pth',
    bg_percentile = 10,
    data_path_input = \
        '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/tif/high_res/20250527/',
    fn_specific_preprocessing = None, # pp_ara.preprocess_getbbox_insideplate2,
    fn_plotting = pp.overlayplot,
    cmap_custom = plt_ara.cmap_custom_plantclasses,
    DPI_plots = 1200
)
```

Using Python's `help(o3.Phase3Config)` function will give you documentation
on the parameters.
An excerpt for the above parameters:

```
    segmentation_dir : str
        Directory where segmentation output will be put. Holds
        segfiles/<subdir>/, plots/<subdir>/, and log_segmentation.yaml.
        <subdir>/ will mimic the original subdirectories of the data input dir.
    nr_classes : int
        Number of different classes (things) to segment.
    nr_channels_input : int
        Type of input, typically 1 for gray scale, and 3 for color images.
    model_checkpoint_to_load : str
        Path to already trained model (.pth file) to be used for segmentation,
        e.g. <training_dir>/models/modelUNet20251026_1027.pth.
    bg_percentile : int
        Determines how images are normalized before shown to ML network.
        The percentile determines what is considered background, which will
        be subtracted to normalize the image intensity range.
    data_path_input : str
        Path to directory with images to segment. May contain subdirectories
        with images.
    fn_specific_preprocessing : Callable | None
        Optional preprocessing function that pre-processes all images to be
        segmented. Should look like:
        `img_toseg_prepr, prepr_info = config.fn_specific_preprocessing(img_toseg)`
        Where `img_toseg` and `img_toseg_prepr` are input and output image,
        `prepr_info` is additional information generated that also gets
        stored later in npz.
    fn_plotting : Callable | None
        If set, plots will be made using this function. Should look like:
        `fig, ax = config.fn_plotting(img, pred, cmap, ..)`
        where **config.extraplottingparams will be passed to the function as well.
    cmap_custom : ListedColormap | None
        Custom cmap of type matplotlib.colors.ListedColormap can be provided
        for predicted segmentation masks. If None, a default cmap will be used.
    DPI_plots : int
        Optional; DPI used for plots.
```

On other machine's than new macbooks, the `target_device` setting is relevant as well.

```
    target_device : str
        Torch device that the model and image tensors are moved to;
        'mps' (Apple Silicon), 'cuda' (NVIDIA) or 'cpu'. Note that 'cpu' will
        typically work on all machines, but will be very slow. Use 'mps' or
        'cuda' if available.
```
