


# U-Net image segmentation

The goal  of this repository is to facilitate easy training and application of an U-net to recognize objects in an image. 

It has been tested and optimized for detection of cells and plants, but can also be used for other purposes.

# Installation

### General requirement, install Conda

To install and use these scripts, you will need 'conda'. This software allows you to
install python and all required libraries.

For instructions on how to install conda, see this [blog post](https://www.biodsc.nl/posts/installing_conda_python.html#how-to-install-conda). It might also be 
convenient to install a Python IDE such as Spyder (software to edit and run python files). 
That is also described in aforementioned link. It is not strictly necessary
though.

### (option A) Install libraries annotation script only

The actual machine learning scripts require the `pytorch` library, which
can sometimes be difficult to install. 
You can also install all libraries required for the scripts, and exclude the machine
learning libraries. 
You cannot use all code in that case, but this might be useful if you e.g. 
only want to run "Phase I" (the annotation part) of the code.

To do this, after you installed conda (rather, 'miniconda'), you need to open a terminal and use the following command:

```bash
conda create -n cheeky-phase1 -c conda-forge numpy pandas scipy scikit-image matplotlib pillow nd2 napari openpyxl pyqt -y
```

### (Option B) Install libraries to use any script within this repository

To use any script in the repository, use the following code:

```bash
conda create -n cheeky-all -c conda-forge -c pytorch python=3.12 numpy pandas matplotlib pytorch torchvision scipy scikit-image napari pillow nd2 seaborn pyqt openpyxl -y
```

On some machines, the libraries `pytorch` and `torchvision` have additional
requirements. If you get an error related to these libraries, you might need
to address these issues by installing additional software.

### Download the scripts

You will need to put the scripts in this repository on your computer.

You can download them using the green "<> Code" button at the 
right top, and press "download zip".

A more advanced option is to set up git and clone the repository to 
your local computer, see below.

```bash
# Navigate to the directory you'd like to install the scripts
cd /path/to/your/directory

# Clone the repository using git
git clone git@github.com:Jintram/Cheeky-cells.git
cd Cheeky-cells
```

Then, you need to run the following command (replace `/path/to/script/directory`
by the actual path):

```bash
pip install -e /path/to/script/directory
```

### Run a script

Afterwards, to run scripts, use:

```bash
# Activate the environment
conda activate cheeky-all

# Run a particular part of the code 
python pipelineclean_phase1_example_roots.py
```

In case you choose the limited installation (option A above), replace
`cheeky-all` with `cheeky-phase1`.

### Run a script from the IDE

In some cases it's convenient to run the script from an IDE. 
In that case, make sure to install the IDE in your conda environment, 
e.g. for Spyder:

```bash
conda activate cheeky-all 
conda install -c conda-forge spyder -y
```

Then, open your IDE from the terminal after activating the Conda environment.
For Spyder, see this example:

```bash
conda activate cheeky-all 
spyder
```

# Training a model

### Phase 1: Annotation of training data

Phase 1 creates a training dataset by letting you manually annotate images using [Napari](https://napari.org), an interactive image viewer.

**Step 1: Create a pipeline script**

Copy one of the example scripts (e.g. `pipelineclean_phase1_example_roots.py`) and edit the configuration to match your dataset. The key settings are:

- `inputdirectory` — folder containing your raw images. By default, files are read from this path without being copied.
- `training_dir` — directory where metadata, annotations, training-session logs, and (optionally) staged originals are stored. Phase 2 must be pointed at the same directory.
- `tile_size` — size of the image tile shown for annotation.
- `file_formats` — tuple of image extensions to include, e.g. `('.tif', '.nd2')`.
- `segfn` — an optional function that produces a preliminary segmentation (saves you drawing from scratch).
- `copy_originals` — set to `True` to copy original training data to `training_dir/originals/` (`False` by default)

**Step 2: Generate metadata**

Run the script up to `phase1_setup(config1)`. This scans `inputdirectory` and creates a file called `metadata_imagefiles_autogen.xlsx` in `training_dir`. Open that Excel file, review and edit it (e.g. mark which images to include, assign datasets), then save it under a new name such as `metadata_imagefiles_manual.xlsx`.

**Step 3: Annotate images**

Set `config1.metadatafiles_path` to point to your edited metadata file, then run `phase1_annotate(config1)`. This will open each image tile in Napari for annotation.

**Using Napari for annotation:**

- The image is displayed as a background layer; on top of it a **Labels layer** called "segmentation" is shown with a preliminary segmentation.
- Select the **Labels layer** in the layer list on the left to start editing.
- Use the **paint brush** tool (keyboard shortcut: `2`) to paint labels onto the image. Change the brush size with `[` and `]`.
- Use the **eraser** (keyboard shortcut: `3`) to remove mistakes.
- Use the **fill** tool (keyboard shortcut: `4`) to fill a region.
- Switch label values with the label selector in the top-left. Typically: `0` = background, `1` = object of interest (e.g. cell interior), and higher values for additional classes.
- **Close the Napari window** (click ✕ or Cmd+W) to save that tile's annotation and move to the next image.
- Press **`q`** to quit the annotation loop entirely (the current tile is not saved).

Annotations are saved as `.npy` files in the `humanseg/` subfolder of your `training_dir`.

# Using a previously trained model to get segmented images ("phase 3")


## Loading libraries and setting up

This section describes how a segmentation run is 
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

# Technical information

See the file [documentation_technical.md](documentation_technical.md) for a 
description of the hierarchy of functions, classes and scripts in this 
code base.