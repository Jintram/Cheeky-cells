


# Technical details

*Walkthrough of the pipeline and hierarchy of functions called.*

# Phase 3

<br><br>

User interacts using the orchestrator (`orchestrate_phase3_clean`):

```
import cheeky_cells.orchestrators.orchestrate_phase3_clean as o3
```

<br><br>

Then, pipeline goes:

- `config3 = o3.Phase3Config(..)`
    - User sets up `config3` object
- `config3 = o3.collect_filelist(config3)`
    - collects file list based on image directory (`data_path_input`)
- `o3.segment_all_files(config3)`
    - segments all files and produce the seg files
    - see below for further info
    
<br><br>

**segment_all_files()**
    
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
            
# U-net Model decsription

<font color=red>Text below drafted by Claude Opus, check this!!</font>

*The machine learning side of the package lives in
`cheeky_cells/machine_learning/`, and consists of three parts: the model
definition itself, a dataset class that feeds it training data, and the
train/test loops.*

<br><br>

The three sub-modules, and who uses them:

- `machine_learning/model/` — the U-net itself (`unet_model.py`, `unet_parts.py`)
    - used by phase 2 (training) and phase 3 (inference)
- `machine_learning/datasetclass/` — dataset class + augmentation (`dataset_classes.py`)
    - used by phase 2 only
- `machine_learning/trainer/` — train and test loops (`trainer.py`)
    - used by phase 2 only

(`machine_learning/applying/` currently only holds an empty `__init__.py`;
inference code lives in the phase 3 orchestrator.)

<br><br>

**model/unet_model.py**

Contains the standard U-net, taken from
https://github.com/milesial/Pytorch-UNet
(original paper: Ronneberger, Fischer & Brox, MICCAI 2015).

- `UNet(n_channels, n_classes, bilinear=False)`
    - `n_channels`: nr of input image channels (phase 2/3 use 3, i.e. RGB)
    - `n_classes`: nr of output classes; output is one logit map per class
    - `bilinear`: if `True`, upsampling uses interpolation instead of
    transposed convolutions (fewer parameters); default `False`
- Layers created in `__init__()`, encoder (contracting) path:
    - `inc` = `DoubleConv(n_channels, 64)`
    - `down1`..`down4` = `Down()`, channels 64→128→256→512→1024
    - each `Down` halves the spatial resolution
- Layers, decoder (expanding) path:
    - `up1`..`up4` = `Up()`, channels 1024→512→256→128→64
    - each `Up` doubles the spatial resolution again, and receives the
    matching encoder output as skip connection
    - `outc` = `OutConv(64, n_classes)`, a 1x1 convolution mapping to
    the per-class logits
- `forward(x)`
    - runs `x1=inc(x)`, then `x2..x5` through the `down*` layers
    - then walks back up, each time passing the corresponding
    encoder tensor: `up1(x5, x4)`, `up2(x, x3)`, `up3(x, x2)`, `up4(x, x1)`
    - RETURNS logits (not probabilities) — so use
    `argmax` over the class dimension to get a segmentation
- `use_checkpointing()`
    - wraps all blocks in `torch.utils.checkpoint` to trade compute for
    memory; not used by the pipeline

Note that the network is fully convolutional, so the input image size is
not fixed by the architecture (though it should be divisible by 16 to
match the four down/up steps).

<br><br>

**model/unet_parts.py**

The building blocks used by `UNet`, all `nn.Module` subclasses:

- `DoubleConv(in_channels, out_channels, mid_channels=None)`
    - (3x3 conv => batchnorm => ReLU) applied twice
    - padding=1, so spatial size is preserved
- `Down(in_channels, out_channels)`
    - `nn.MaxPool2d(2)` followed by a `DoubleConv`
- `Up(in_channels, out_channels, bilinear=True)`
    - upsamples `x1` (either `nn.Upsample` or `nn.ConvTranspose2d`,
    depending on `bilinear`)
    - pads `x1` so it matches the skip-connection tensor `x2` in size
        - (needed when input dimensions are not neatly divisible)
    - concatenates `x2` and `x1` along the channel dimension, then
    applies a `DoubleConv`
- `OutConv(in_channels, out_channels)`
    - single 1x1 convolution, no activation

<br><br>

**datasetclass/dataset_classes.py**

Provides the PyTorch `Dataset` that serves training tiles, plus the
augmentation pipelines and label-weighting helpers.

- `ImageDataset_tiles(df_metadata, datadir, train_or_test, ...)`
    - a `torch.utils.data.Dataset`; images and labels are `.npy` files in
    `datadir`, named after `df_metadata['filename']` plus a suffix
    - `__init__()`
        - selects the rows of `df_metadata` where
        `train_or_test` matches the requested set
        - builds `filelist_imgs` and `filelist_labels` using
        `crw.addsuffixtofilenames()` with `img_suffix` / `lbl_suffix`
        - `len_augment` = `max(nr of files, ARTIFICIAL_N)`; this is the
        length the dataset reports, such that augmentation can generate
        more samples than there are files
    - `__len__()` RETURNS `len_augment`
    - `__getitem__(idx)`
        - maps `idx` onto an actual file via `idx % num_samples`, so all
        underlying files are sampled evenly
        - loads image and label with `np.load()`
        - casts int64 labels to int32 (`PIL.Image.fromarray` cannot
        handle int64)
        - draws one random seed and sets `torch.manual_seed()` twice, once
        before transforming the image and once before transforming the
        label, so **both get the identical random transformation**
        - RETURNS (image, label), both moved to `targetdevice`
        - if no transform was given, it warns and returns the raw arrays
        as tensors
- `augmentation_pipeline_input` — `transforms.Compose` applied to the image
    - convert to PIL => random horizontal flip => random vertical flip =>
    random rotation (±45°) => random 500x500 crop => `ToTensor()`
    - note the crop is done last on purpose; cropping first would produce
    many sliced objects with poorly annotated edges
- `augmentation_pipeline_label` — same geometric operations, but ends in a
`LongTensor` conversion instead of `ToTensor()` (labels are class indices,
they must not be rescaled)
- `get_label_frequencies(dataset, suffix_img, suffix_lbl)`
    - loops over all label files of the dataset, `np.bincount()`s them,
    zero-pads to equal length and sums
    - RETURNS per-class pixel counts
- `get_label_weights(dataset, suffix_img, suffix_lbl, device='mps')`
    - RETURNS the inverse of the relative class frequencies as a tensor
    - used in phase 2 to build a class-weighted `CrossEntropyLoss`, so that
    rare classes are not ignored by the model

<br><br>

**trainer/trainer.py**

Contains the two per-epoch loops; the epoch loop itself and the
optimizer/scheduler/loss live in the phase 2 orchestrator
(`o2.train_model()`).

- `train_loop(dataloader, model, loss_fn, optimizer, dataset_len, BATCH_SIZE, train_log_interval=10)`
    - sets `model.train()` (activates dropout and batchnorm updates)
    - per batch: `pred = model(X)`, `loss = loss_fn(pred, y)`, then
    `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`
    - records and prints the loss every `train_log_interval` batches
    - RETURNS `loss_tracker`, the list of sampled losses
- `test_loop(dataloader, model, loss_fn, dataset_len, BATCH_SIZE, nr_classes=None)`
    - sets `model.eval()` and runs under `torch.no_grad()`
    (no gradients, less memory)
    - accumulates the test loss, and the accuracy as the fraction of
    correctly predicted pixels (`y == pred.argmax(1)`)
    - accumulates a confusion matrix over all pixels using `np.add.at()`
        - `nr_classes` sets its size; if not given it is taken from the
        prediction's channel dimension
    - both loss and accuracy are averaged over the nr of batches
    - RETURNS (correct, test_loss, confusion_matrix)

<br><br>

**How phase 2 wires this together**

- `o2.create_datasets()` builds a train and a test `ImageDataset_tiles`,
both with `augmentation_pipeline_input` / `augmentation_pipeline_label`
- `o2.initialize_unet_model()` creates `cunet.UNet(n_channels=3,
n_classes=config2.nr_classes)` and moves it to `config2.target_device`
- `o2.train_model()`
    - builds `CrossEntropyLoss` weighted by `cdc.get_label_weights()`
    - builds an `Adam` optimizer and a `LambdaLR` scheduler driven by
    `config2.lr_schedule_relative` (via `o2.custom_lr_schedule()`)
    - wraps both datasets in a `DataLoader`
    - loops over `config2.epochs`, each epoch calling `ct.train_loop()`
    and then `ct.test_loop()`, and stepping the scheduler
- phase 3 only needs `cunet.UNet()` plus `model.load_state_dict()`;
the dataset class and trainer are not involved (see above)
