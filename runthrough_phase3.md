


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

The name `config3_ara_root` is arbitrary. In this example, the 3 refers to phase 3,
and `ara_root` to the type of data we're processing.

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

#### Note to self, things to improve

*The parameter names `segmentation_dir` and `data_path_input` could have
more clear names. E.g. `directory_input_images` and `directory_segmentation_output`.*

### Collecting list of image files to segment.

To continue segmentation, the pipeline requires you to collect a list of images,
which can be done with 

```
config3_ara_root = o3.collect_filelist(config3_ara_root)
```

this will store a file list into the configuration object. 

Optionally, you can take a look at the data, 

```
config3_ara_root.df_metadata
```

Yields:

```
	subdir	filename	segmentation_channel	train_or_test
0	.	20250530_OY_07.tif	all	
1	.	20250527_OY05.tif	all	
2	.	20250527_OY11.tif	all	 
(..)
```

The `segmentation_channel` and `train_or_test` are for advanced purposes, ie in case
you want to re-use this data for training.

For a general segmentation run, `<yourconfig>.df_metadata` just serves as a file list (in pandas dataframe format).

## Description of the pipeline itself

Running the command `o3.segment_all_files(<yourconfig>)` will 
now automatically start segmenting the images in the folder 
set by `<yourconfig>.data_path_input`.

Additional options to the `o3.segment_all_files()` function are
`max_files_to_process` and `overwrite_files=True`, as can be found with 
`help(o3.segment_all_files)`.

```    
overwrite_files: 
    Boolean indicating whether to overwrite existing segmentation files.
max_files_to_process: 
    Maximum number of files to process. If None, process all files. 
    Intended for testing purposes.
```

### Pipeline output

The pipeline produces a directory structure as follows:

```
<output-dir>
    plots/        
        contains a plot of each segmented file (if fn_plotting is set)
        the plots show the segmentation result for visual inspection.
        subdirectory structure follows that of original data.
    segfiles/
        contains _seg.npz files that contain the segmentation information.        
    log_segmentation.yaml
```

This is the endpoint of this segmentation pipeline.

The results in the npz files can be read by other scripts for further 
processing.

# Technical details

Walkthrough of the pipeline and functions called:

```
import cheeky_cells.orchestrators.orchestrate_phase3_clean as o3
```

- `config3 = o3.Phase3Config(..)`
    - User sets up `config3` object
- `config3 = o3.collect_filelist(config3)`
    - collects file list based on image directory (`data_path_input`)
- `o3.segment_all_files(config3)`
    - segments all files
    
- `o3.segment_all_files(config3)` 
    - imports 
    ```
    import cheeky_cells.readwrite.cheeky_readwrite as crw
    import cheeky_cells.annotating_data.annotation_aided as caa
    from cheeky_cells.machine_learning.model import unet_model as cunet
    ```
    - blabla