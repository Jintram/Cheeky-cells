


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
    - segments all files and produce the seg files
    - see below for further info
    
<br><br>
    
- `o3.segment_all_files(config3)` 
    - imports 
    ```
    import cheeky_cells.readwrite.cheeky_readwrite as crw
    import cheeky_cells.annotating_data.annotation_aided as caa
    from cheeky_cells.machine_learning.model import unet_model as cunet
    ```
    - calls:
        - `initialize_unet_model_for_inference()` 
            - sets up U-net using `cunet.UNet()`
            - (one liner)
        - `load_model_checkpoint`
            - loads model using `model_unet.load_state_dict`
                - (this is a default PyTorch function)
            - (one liner)
    - Loops over each file, and calls:
        - `get_input_img_file()` which calls 
            - `crw.loadimgfile_metadata()` to load an image
            based on the file list and current file index.
            - `config.fn_specific_preprocessing()` (if available)
            to preprocess image (e.g. cropping)
            - `crw.image_autorescale()` to scale the image.
            - RETURNS loaded image, rescaled image, extra info
        - `get_ml_prediction()` which 
            - performs tensor conversion
            - calls `the_model(X)` yielding the prediction (as tensor)
            - converts prediction to numpy array
        - if `config.fn_plotting` is set
            - creates plot in output dir using `fn_plotting()`
        - saves the prediction
            - using `np.savez_compressed()` and `np.save()`
        - each loop, Torch cache is reset to prevent memory build-up
            - (`torch.cuda.empty_cache()` or `torch.mps.empty_cache()`