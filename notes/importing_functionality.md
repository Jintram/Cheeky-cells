

## Importing functionalities

The goal of this code base is to be generically applicable to all kinds of
segmentation problems.

However, for some projects it is convenient to have data-specific functionalities
that allow you to perform certain actions.
For example, when annotating plants, it's convenient to have specific 
editor options geared towards plants.

Therefor, you can, instead of using local functions, import other functions,
and use those. This involves two steps.

### (1) import the repository that holds the external function

```bash
# to make the other repo available; -e signals that editable mode
# (otherwise files get copied in current state to env directory)
pip install -e /path/to/other/repo
```

e.g. 

```
conda activate 2025_pytorch2
pip install -e /Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length/
```

*Technical note: for this to work the repository that gets imported, needs
to have a `pyproject.toml` file and `__init__.py` files in the relevant subfolders.*


### (2) import the specific function and 

Then, if there's an available slot, a function can be saved to a `config`
object.

E.g.

```
import root_length.functions_pipeline.edit_segfiles as pl_edit
config1.my_napari_function = pl_edit.edit_annotation_napari
```

### A technical note

This approach is not used consistently throughout the code. There are also
project-specific pieces of code still in this repo. E.g. with

```
import annotating_data.dedicated_segmentation as cds
segfn = cds.basicplantseg1
```

`basicplantseg1()` is a function that is defined within this repository.



