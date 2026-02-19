


## Changelog file


### 2026-02-18

I started this changelog file to replace notes I stored digitally elsewhere.

#### Current aims

**Main aim:** The final aim of this repo is that a user can provide any set of images with ground truths, to train a network, extract the model, and apply it to a new set of images.

**Sub aim:** Get the pipeline running such that it produces segmentation files in a structured way for the plant data.

**Current code status:** The code is now quite functionalized (the last part being done with some help of ChatGPT Codex 5.3), and rather modular. 
Not all functions are in the right place however. 
Architectually, the code consist of three parts that I consider separate parts of the procedure. These are listed below, together with where code that it uses is located. The newest implementation of the code is only available for the arabidopsis root example.
1. Create the training set.
    - High-level orchestration functions: 
        - `pipeline_example_arabidopsis_ML_phase1_create_training_set.py`
    - Related submodules:
        -  `readwrite.cheeky_readwrite` (aka `crw`)
        -  `annotating_data.annotation_aided` (aka `caa`)
        -  `annotating_data.dedicated_segmentation` (aka `cds`)
    - Status:
        - This functions more or less as intended;
            - Functions are located in submodules
            - Submodules are in logical places
2. Train the model.
    - Status: high level file needs to be compressed, and refactored into submodules in right place --> probably `machine_learning/trainer/trainer.py` is a logical place.
    - High level orchestration functions:
        - `pipeline_example_arabidopsis_ML_phase2_train_model.py`
            - **modification required:** The goal is that this file is high-level, but it contains quite a load of detailed functions. Needs to be moved to submodules. See functions starting at `custom_lr_schedule`.
                - There's also some miscelaneous functions that generate sanity-checking output, have to think of a place for those.
        - Related submodules:
            - `machine_learning.datasetclass.dataset_classes` (aka `cdc`)
            - `machine_learning.model.unet_model` (aka `cunet)`
            - `machine_learning.trainer.trainer` (aka `ct`)
3. Apply the model.
    - High level orchestrator function: `pipeline_example_arabidopsis_ML_phase3_apply_model.py`
        - **Modifications required;** 
            - Currently, this file contains functions that are used to apply the ML model. These functions should be moved to `machine_learning/apply` or something similar. 
            - The subfunctions are also a bit poorly organized, with both functions to show predictions for images loaded with the loader, and actual images being processed. 
    - Related submodules
        - *Related submodules are not really existing, only some functions from previous submodules are loaded to be used agian.*

#### Planned changes

- Generalize the files:
    - `pipeline_example_arabidopsis_ML_phase1_create_training_set.py`
    - `pipeline_example_arabidopsis_ML_phase2_train_model.py`
    - `pipeline_example_arabidopsis_ML_phase3_apply_model.py`
- Create submodules in the `machine_learning` subdir framework for functions that are too detailed for the high-level `pipeline_.._phase..` files.
- Add new arabidopsis root-specific pipeline example file, which uses the high-level files, setting new configuration (probably remove the current default arabidopsis root settings).

###### Current focus

- Generate a pipeline that will properly 
    - take existing input data and process it into segmented files
    - whilst retaining metadata information
    
What is the best way to (in a generalized way) process metadata information?

**Issue to address:** arabidopsis root data doesn't have any annotation for conditions; so will simply retain subfolder and filenames into column.

Anyways this will allow for a later generalized strategy, since even if filenames contain metadata, this can be post-processed in a dataframe.

PERHAPS IT IS NOT NECESSARY TO WORRY ABOUT METADATA AT THIS STAGE, AS FILES CAN BE 1:1 CONVERTED TO `_seg.npz` FILES.

**Current actions:** 
- MAKE A NEW FILE `orchestrate_phase3_clean.py`.
    - based on `pipeline_example_arabidopsis_ML_phase3_apply_model.py`
- Find out how it is decided which files are processed.
- Reshape this such that a symmetric folder structure is created which holds seg-files


    