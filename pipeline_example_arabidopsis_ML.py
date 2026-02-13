# %% ################################################################################
# Functionalized Arabidopsis ML pipeline
#
# This script has three parts:
# 1) creating a training set for an AI segmentation model
# 2) training the model
# 3) applying the model to process actual data

from dataclasses import dataclass
import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import LambdaLR


# %% ################################################################################
# Configuration container

@dataclass
class PipelineConfig:
    # Root directory of this repository
    script_dir: str = os.path.dirname(os.path.abspath(__file__))

    # Directory with source image files for building training metadata
    basedirectory: str = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/'
    # Subdirectories with files used for metadata generation
    subdirs: tuple = ('SELECTION_ML/Cropped/',)

    # Directory for all output from this script
    outputdirectory: str = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/'

    # Optional: set True when you want to regenerate metadata file
    regenerate_training_metadata: bool = False
    # Manually edited metadata file used for training tiles
    metadata_customized_filename: str = 'metadata_imagefiles_manual20251022.xlsx'

    # Segmentation + training settings
    tile_size: int = 1000
    bg_percentile: int = 10
    nr_classes: int = 5
    target_device: str = 'mps'

    # Dataset settings
    img_suffix: str = '_tile_img_enhanced'
    lbl_suffix: str = '_tile_seg'
    artificial_n: int = 1000
    crop_size: int = 1000

    # Training settings
    learning_rate: float = 1e-3
    batch_size: int = 8
    epochs: int = 24
    lr_schedule_step_len: int = 8

    # Optional pretrained checkpoint to load
    model_checkpoint_to_load: str | None = None

    # Input data settings for applying model
    data_path_input: str = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/'
    regenerate_input_metadata: bool = True
    metadata_input_filepath: str = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/metadata_imagefiles_manual_inputdata20251027.xlsx'

    # Demo path for single-image prediction
    image_path_test_single: str = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/20250721batch1/20250721_OY_09.jpg'

    # Default pipeline toggles
    run_phase1_create_training_set: bool = True
    run_phase2_train_model: bool = True
    run_phase3_apply_model: bool = True

    # How many random full images to process in phase 3
    n_random_input_files: int = 100


# %% ################################################################################
# Path + environment helpers


def get_output_paths(config: PipelineConfig) -> dict:
    # Build output folder paths in one place
    paths = {
        'segfolder': os.path.join(config.outputdirectory, 'humanseg/'),
        'modelfolder': os.path.join(config.outputdirectory, 'models/'),
        'pltfolder': os.path.join(config.outputdirectory, 'plots/'),
    }
    return paths


def ensure_output_dirs(paths: dict) -> None:
    # Create needed output folders
    os.makedirs(paths['segfolder'], exist_ok=True)
    os.makedirs(paths['modelfolder'], exist_ok=True)
    os.makedirs(paths['pltfolder'], exist_ok=True)


def add_script_dir_to_path(script_dir: str) -> None:
    # Add repository root to Python path when needed
    if script_dir not in sys.path:
        sys.path.append(script_dir)


def resolve_device(requested_device: str) -> str:
    # Keep original default behavior (mps), but fall back to cpu if unavailable
    if requested_device == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# %% ################################################################################
# Import custom modules after path handling


def import_custom_modules():
    # Import custom modules used across all phases
    import readwrite.cheeky_readwrite as crw
    import annotating_data.annotation_aided as caa
    import annotating_data.annotation_postprocessing as cap
    import annotating_data.dedicated_segmentation as cds
    import prepostprocessing_input.preprocessing_input as cpp
    from machine_learning.datasetclass import dataset_classes as cdc
    from machine_learning.model import unet_model as cunet
    from machine_learning.trainer import trainer as ct

    return crw, caa, cap, cds, cpp, cdc, cunet, ct


# %% ################################################################################
# Shared plotting helpers


def get_plant_cmap() -> ListedColormap:
    # Custom color map for class labels
    custom_colors_plantclasses = [
        '#000000',  # background
        '#90EE90',  # shoot
        '#FFFFFF',  # root
        '#A52A2A',  # seed
        '#006400',  # leaf
    ]
    return ListedColormap(custom_colors_plantclasses)


def sidebysideplot(current_img_rgb, current_prd, current_lbl, plot_name: str, pltfolder: str, cmap_plantclasses) -> None:
    # Compare input / prediction / truth side by side
    cm_to_inch = 1 / 2.54
    fig, ax = plt.subplots(1, 3, figsize=(15 * cm_to_inch, 5 * cm_to_inch))
    ax[0].imshow(current_img_rgb)
    ax[1].imshow(current_prd[0].argmax(0), cmap=cmap_plantclasses, vmin=0, vmax=5)
    ax[2].imshow(current_lbl, cmap=cmap_plantclasses, vmin=0, vmax=5)
    for idx_ax in range(3):
        ax[idx_ax].set_xticks([])
        ax[idx_ax].set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(pltfolder, f'prediction_{plot_name}.pdf'), dpi=300)
    plt.close()


def overlayplot(current_img_rgb, current_prd, current_lbl, plot_name: str, pltfolder: str, cmap_plantclasses) -> None:
    # Overlay predicted labels on top of RGB input image

    # Convert logits to label image when needed
    if current_prd.ndim > 2:
        current_pred_lbl = current_prd[0].argmax(0)
    else:
        current_pred_lbl = current_prd

    # Add a small left padding (kept from original script)
    current_pred_lbl_transl = np.pad(current_pred_lbl, ((0, 0), (20, 0)), 'constant', constant_values=0)

    # Keep image ratio in figure size
    img_shape_ratio = current_img_rgb.shape[1] / current_img_rgb.shape[0]

    # Plot with or without truth panel
    cm_to_inch = 1 / 2.54
    if current_lbl is None:
        fig, ax = plt.subplots(1, 1, figsize=(10 * img_shape_ratio * cm_to_inch, 10 * cm_to_inch))
        ax.imshow(current_img_rgb)
        ax.imshow(current_pred_lbl_transl, cmap=cmap_plantclasses, vmin=0, vmax=5, alpha=1.0 * (current_pred_lbl_transl > 0))
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig, ax = plt.subplots(1, 2, figsize=((5 * img_shape_ratio + 5) * cm_to_inch, 5 * cm_to_inch))
        ax[0].imshow(current_img_rgb)
        ax[0].imshow(current_pred_lbl_transl, cmap=cmap_plantclasses, vmin=0, vmax=5, alpha=1.0 * (current_pred_lbl_transl > 0))
        ax[1].imshow(current_lbl, cmap=cmap_plantclasses, vmin=0, vmax=5)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    # Save file
    plt.tight_layout()
    plt.savefig(os.path.join(pltfolder, f'predictionoverlay_{plot_name}.pdf'), dpi=300)
    plt.close()


# %% ################################################################################
# Phase 1: Create training data


def load_training_metadata(config: PipelineConfig, crw) -> pd.DataFrame:
    # Optionally generate metadata first
    if config.regenerate_training_metadata:
        crw.gen_metadatafile(config.basedirectory, list(config.subdirs), config.outputdirectory)

    # Load manually edited metadata
    metadata_filepath = os.path.join(config.outputdirectory, config.metadata_customized_filename)
    df_metadata = pd.read_excel(metadata_filepath)
    return df_metadata


def create_training_set_from_metadata(df_metadata: pd.DataFrame, config: PipelineConfig, output_paths: dict, caa, cds) -> None:
    # Generate assisted annotations and tiles
    caa.annotate_all_pictures_aided(
        df_metadata,
        output_segfolder=output_paths['segfolder'],
        intitial_segfolder=None,
        TILE_SIZE=config.tile_size,
        tile_selection_by='maxvar',
        segfn=cds.basicplantseg1,
        ignore_saved_file=False,
        showplots=False,
        rescalelog=False,
        bg_percentile=config.bg_percentile,
        showrawimg=True,
    )


# %% ################################################################################
# Phase 2: Train model


def build_train_test_datasets(df_metadata: pd.DataFrame, config: PipelineConfig, output_paths: dict, cdc):
    # Create train dataset
    dataset_train = cdc.ImageDataset_tiles(
        df_metadata,
        output_paths['segfolder'],
        'train',
        img_suffix=config.img_suffix,
        lbl_suffix=config.lbl_suffix,
        transform=cdc.augmentation_pipeline_input,
        transform_label=cdc.augmentation_pipeline_label,
        targetdevice=config.target_device,
        ARTIFICIAL_N=config.artificial_n,
        CROP_SIZE=config.crop_size,
    )

    # Create test dataset
    dataset_test = cdc.ImageDataset_tiles(
        df_metadata,
        output_paths['segfolder'],
        'test',
        img_suffix=config.img_suffix,
        lbl_suffix=config.lbl_suffix,
        transform=cdc.augmentation_pipeline_input,
        transform_label=cdc.augmentation_pipeline_label,
        targetdevice=config.target_device,
        ARTIFICIAL_N=config.artificial_n,
        CROP_SIZE=config.crop_size,
    )

    return dataset_train, dataset_test


def initialize_unet_model(config: PipelineConfig, cunet):
    # Initialize U-net with selected class count
    model_unet = cunet.UNet(n_channels=3, n_classes=config.nr_classes).to(config.target_device)
    return model_unet


def custom_lr_schedule(epoch: int, step_len: int = 50):
    # Return LR scale factor for each epoch
    lr_scalefactor = [1] * step_len + [0.1] * step_len + [0.01] * step_len
    if epoch >= step_len * 3:
        epoch = step_len * 3 - 1
    return lr_scalefactor[epoch]


def train_model(dataset_train, dataset_test, model_unet, config: PipelineConfig, cdc, ct):
    # Build class-weighted loss
    label_weights = cdc.get_label_weights(dataset_train, suffix_img=config.img_suffix, suffix_lbl=config.lbl_suffix)
    loss_fn = torch.nn.CrossEntropyLoss(label_weights)

    # Build optimizer
    optimizer = torch.optim.Adam(model_unet.parameters(), config.learning_rate)

    # Build loaders
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=True)

    # Build scheduler
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: custom_lr_schedule(epoch, step_len=config.lr_schedule_step_len),
    )

    # Run training loop
    list_loss_tracker = []
    list_correct = []
    start_time_overall = time.time()

    for epoch_idx in range(config.epochs):
        print('=' * 30)
        print(f"Epoch {epoch_idx + 1}, LR: {scheduler.get_last_lr()}")
        start_time = time.time()

        # Train and validate
        loss_tracker = ct.train_loop(
            train_loader,
            model_unet,
            loss_fn,
            optimizer,
            len(dataset_train),
            config.batch_size,
        )
        current_correct = ct.test_loop(
            val_loader,
            model_unet,
            loss_fn,
            len(dataset_test),
            config.batch_size,
        )

        # Update scheduler
        scheduler.step()

        # Store metrics
        list_loss_tracker.append(loss_tracker)
        list_correct.append(current_correct)

        # Print duration for this epoch
        elapsed_time = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - start_time))
        print(f"Epoch {epoch_idx + 1} completed in {elapsed_time}..")

    # Print overall duration
    elapsed_time_overall = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - start_time_overall))
    print(f'Training completed in {elapsed_time_overall}..')
    print('Done!')

    return model_unet, list_loss_tracker, list_correct


def save_model_checkpoint(model_unet, output_paths: dict) -> str:
    # Save model with timestamp
    current_time_formatted = time.strftime('%Y%m%d_%H%M')
    output_path = os.path.join(output_paths['modelfolder'], f'modelUNet{current_time_formatted}.pth')
    torch.save(model_unet.state_dict(), output_path)
    return output_path


def load_model_checkpoint(model_unet, checkpoint_path: str):
    # Load model checkpoint if provided
    model_unet.load_state_dict(torch.load(checkpoint_path))
    return model_unet


def plot_training_history(list_loss_tracker, list_correct, epochs: int) -> None:
    # Flatten batch-loss history
    datay = np.array(list_loss_tracker).flatten()
    datax = np.array(range(len(datay))) * 100

    # Plot train loss
    plt.plot(datax, datay, color='black')
    plt.ylim([0, np.max(datay) * 1.1])
    plt.axvline(datax[-1] / 3, linestyle='--', color='black')
    plt.axvline(datax[-1] / 3 * 2, linestyle='--', color='black')

    # Plot test correctness
    data_correctx = np.linspace(datax[-1] / epochs, datax[-1], epochs)
    data_correcty = np.array(list_correct)
    plt.plot(data_correctx, data_correcty)

    # Show figure
    plt.show()
    plt.close()


def evaluate_on_tiles_and_plot(model_unet, dataset_train, dataset_test, output_paths: dict, cmap_plantclasses, n_examples: int = 10):
    # Evaluate a few train and test samples and save plots
    for whichone in ['test', 'train']:
        for idx in range(n_examples):
            # Select sample
            if whichone == 'train':
                current_img, current_lbl = dataset_train[idx]
            else:
                current_img, current_lbl = dataset_test[idx]

            # Predict
            X = current_img[None, :, :, :]
            logits = model_unet(X)

            # Prepare arrays for plotting
            current_img = current_img.cpu().numpy()
            current_img_rgb = current_img.transpose(1, 2, 0)
            current_lbl = current_lbl.cpu().numpy()
            current_prd = logits.cpu().detach().numpy()

            # Compute pixel accuracy
            correct_pixels = current_prd[0].argmax(0) == current_lbl
            accuracy = np.sum(correct_pixels) / correct_pixels.size
            print(f'Accuracy for {whichone} image {idx}: {accuracy * 100:.2f} %')

            # Build stable plot name
            plot_name = f'{whichone}_{idx:03d}'

            # Save side-by-side and overlay plots
            sidebysideplot(current_img_rgb, current_prd, current_lbl, plot_name, output_paths['pltfolder'], cmap_plantclasses)
            overlayplot(current_img_rgb, current_prd, current_lbl, plot_name, output_paths['pltfolder'], cmap_plantclasses)


# %% ################################################################################
# Phase 3: Apply model


def read_input_img_file(df_metadata_input: pd.DataFrame, file_idx: int, crw, cpp, caa, bg_percentile: int):
    # Read one image from metadata
    img_test = crw.loadimgfile_metadata(df_metadata_input, file_idx)

    # Crop to plate interior
    img_test_crop, _ = cpp.preprocess_getbbox_insideplate(img_test)

    # Normalize intensity
    img_test_crop_norm = caa.image_autorescale(img_test_crop, rescalelog=False, bg_percentile=bg_percentile)
    return img_test_crop_norm


def get_ml_prediction(img_input, the_model, target_device: str, showplot: bool, cmap_plantclasses):
    # Run single prediction
    with torch.no_grad():
        img_torch = ToTensor()(img_input).to(target_device)
        X = img_torch[None, :, :, :]
        logits = the_model(X)
        prd_full = logits.cpu().detach().numpy()
        prd_labels = prd_full[0].argmax(0)

    # Optional plot for quick check
    if showplot:
        plt.imshow(prd_labels, cmap=cmap_plantclasses, vmin=0, vmax=4)
        plt.show()

    return prd_labels


def load_input_metadata(config: PipelineConfig, crw) -> pd.DataFrame:
    # Optionally regenerate metadata for input data folder
    if config.regenerate_input_metadata:
        all_subfolders = glob.glob(pathname='*', root_dir=config.data_path_input)
        crw.gen_metadatafile(
            config.data_path_input,
            subdirs=all_subfolders,
            outputdirectory=config.outputdirectory,
            suffix='_inputdata',
            default_columns=['comments'],
        )

    # Load selected metadata file
    df_metadata_input = pd.read_excel(config.metadata_input_filepath)
    return df_metadata_input


def apply_model_to_single_image(config: PipelineConfig, model_unet, caa, cmap_plantclasses, output_paths: dict):
    # Read single image from absolute path
    from PIL import Image

    img_full = np.array(Image.open(config.image_path_test_single).convert('RGB'))

    # Normalize image
    img_full_norm = caa.image_autorescale(img_full, rescalelog=False, bg_percentile=config.bg_percentile)

    # Predict full image
    with torch.no_grad():
        img_full_torch = ToTensor()(img_full_norm).to(config.target_device)
        X = img_full_torch[None, :, :, :]
        logits = model_unet(X)
        prd_full = logits.cpu().detach().numpy()

    # Save overlay plot
    current_img = X[0].cpu().numpy()
    current_img_rgb = current_img.transpose(1, 2, 0)
    overlayplot(
        current_img_rgb,
        prd_full,
        None,
        plot_name='fullimage_test',
        pltfolder=output_paths['pltfolder'],
        cmap_plantclasses=cmap_plantclasses,
    )


def apply_model_to_random_input_batch(df_metadata_input: pd.DataFrame, config: PipelineConfig, model_unet, crw, cpp, caa, cmap_plantclasses, output_paths: dict):
    # Build output folder for full-image predictions
    output_prediction_folder = os.path.join(output_paths['pltfolder'], 'fullimages_predictions/')
    os.makedirs(output_prediction_folder, exist_ok=True)

    # Pick random file indices
    random_file_idxs = np.random.choice(df_metadata_input.index, size=config.n_random_input_files, replace=False)

    # Loop over selected files
    for file_idx in random_file_idxs:
        print(f'Processing file idx {file_idx} ..')

        # Read and normalize image
        img_test_crop_norm = read_input_img_file(
            df_metadata_input,
            file_idx=file_idx,
            crw=crw,
            cpp=cpp,
            caa=caa,
            bg_percentile=config.bg_percentile,
        )

        # Predict labels
        img_pred_lbls = get_ml_prediction(
            img_test_crop_norm,
            the_model=model_unet,
            target_device=config.target_device,
            showplot=False,
            cmap_plantclasses=cmap_plantclasses,
        )

        # Save overlay plot
        print('Plotting prediction')
        overlayplot(
            img_test_crop_norm,
            img_pred_lbls,
            None,
            plot_name=f'fullimage_idx{file_idx:03d}',
            pltfolder=output_prediction_folder,
            cmap_plantclasses=cmap_plantclasses,
        )

        # Save predicted labels as numpy array
        print('Saving prediction..')
        np.save(os.path.join(output_prediction_folder, f'predictedmask_idx{file_idx:03d}.npy'), img_pred_lbls)


# %% ################################################################################
# Default pipeline runner (small function called at bottom)


def run_default_pipeline(config: PipelineConfig):
    # Step 0: prepare paths and imports
    add_script_dir_to_path(config.script_dir)
    output_paths = get_output_paths(config)
    ensure_output_dirs(output_paths)

    # Resolve target device once
    config.target_device = resolve_device(config.target_device)
    print(f'Using target device: {config.target_device}')

    # Import custom modules once after path setup
    crw, caa, cap, cds, cpp, cdc, cunet, ct = import_custom_modules()

    # Build colormap used in all prediction plots
    cmap_plantclasses = get_plant_cmap()

    # Keep references that are needed across phases
    model_unet = None
    df_metadata = None

    # ---------------------------------------------------------------------
    # Part 1: create training set from metadata
    # ---------------------------------------------------------------------
    if config.run_phase1_create_training_set:
        print('\n--- Part 1: creating training set ---')
        df_metadata = load_training_metadata(config, crw)
        create_training_set_from_metadata(df_metadata, config, output_paths, caa, cds)

    # ---------------------------------------------------------------------
    # Part 2: train model
    # ---------------------------------------------------------------------
    if config.run_phase2_train_model:
        print('\n--- Part 2: training model ---')

        # Ensure metadata is loaded when phase 2 runs independently
        if df_metadata is None:
            df_metadata = load_training_metadata(config, crw)

        # Build datasets
        dataset_train, dataset_test = build_train_test_datasets(df_metadata, config, output_paths, cdc)

        # Initialize model
        model_unet = initialize_unet_model(config, cunet)

        # Optional: load checkpoint before continuing
        if config.model_checkpoint_to_load is not None:
            model_unet = load_model_checkpoint(model_unet, config.model_checkpoint_to_load)

        # Train model
        model_unet, list_loss_tracker, list_correct = train_model(
            dataset_train,
            dataset_test,
            model_unet,
            config,
            cdc,
            ct,
        )

        # Save model checkpoint
        saved_model_path = save_model_checkpoint(model_unet, output_paths)
        print(f'Saved checkpoint: {saved_model_path}')

        # Plot training history
        plot_training_history(list_loss_tracker, list_correct, config.epochs)

        # Plot prediction quality on train and test tiles
        evaluate_on_tiles_and_plot(
            model_unet,
            dataset_train,
            dataset_test,
            output_paths,
            cmap_plantclasses,
            n_examples=10,
        )

    # ---------------------------------------------------------------------
    # Part 3: apply model to actual data
    # ---------------------------------------------------------------------
    if config.run_phase3_apply_model:
        print('\n--- Part 3: applying model ---')

        # Build model when part 3 is run without part 2
        if model_unet is None:
            model_unet = initialize_unet_model(config, cunet)

            # For phase-3-only runs, checkpoint should be provided
            if config.model_checkpoint_to_load is None:
                raise ValueError('For phase-3-only runs, set config.model_checkpoint_to_load to a valid .pth file.')

            model_unet = load_model_checkpoint(model_unet, config.model_checkpoint_to_load)

        # Run one single-image demo prediction
        apply_model_to_single_image(config, model_unet, caa, cmap_plantclasses, output_paths)

        # Load metadata for input folder and run random-batch predictions
        df_metadata_input = load_input_metadata(config, crw)
        apply_model_to_random_input_batch(
            df_metadata_input,
            config,
            model_unet,
            crw,
            cpp,
            caa,
            cmap_plantclasses,
            output_paths,
        )


# %% ################################################################################
# Run the default pipeline

if __name__ == '__main__':
    # Create default config with all settings
    default_config = PipelineConfig()

    # Run the full default pipeline
    run_default_pipeline(default_config)
