


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

# Use of the script

### Phase 1: Annotation of training data

Phase 1 creates a training dataset by letting you manually annotate images using [Napari](https://napari.org), an interactive image viewer.

**Step 1: Create a pipeline script**

Copy one of the example scripts (e.g. `pipelineclean_phase1_example_roots.py`) and edit the configuration to match your dataset. The key settings are:

- `inputdirectory` — folder containing your raw images.
- `outputdirectory` — folder where metadata and annotations will be saved.
- `tile_size` — size of the image tile shown for annotation.
- `file_formats` — tuple of image extensions to include, e.g. `('.tif', '.nd2')`.
- `segfn` — an optional function that produces a preliminary segmentation (saves you drawing from scratch).

**Step 2: Generate metadata**

Run the script up to `phase1_setup(config1)`. This scans `inputdirectory` and creates a file called `metadata_imagefiles_autogen.xlsx` in the output directory. Open that Excel file, review and edit it (e.g. mark which images to include, assign datasets), then save it under a new name such as `metadata_imagefiles_manual.xlsx`.

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

Annotations are saved as `.npy` files in the `humanseg/` subfolder of your output directory.

